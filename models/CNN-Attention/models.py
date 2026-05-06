import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_models import AttentionBlock


# ======================================================================
# Building blocks
# ======================================================================

class CausalConv1d(nn.Module):
    """Conv1d with left-only (causal) padding to preserve sequence length."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0, bias=bias
        )

    def forward(self, x):
        # x: (N, C, L)
        x = F.pad(x, (self.pad_len, 0))
        return self.conv(x)


class ConvBlock(nn.Module):
    """
    Equivalent to Conv_block_() in the original Keras code.

    Expects input shape (N, 1, T, in_chans) — PyTorch channels-first 2-D.

    Internally:
      conv1  : temporal Conv2d  (kern_length × 1),  1 → F1 filters
      dw_conv: depthwise Conv2d (1 × in_chans),      F1 → F2=F1*D filters,
               collapses the sensor-channel (W) dimension to 1
      conv3  : pointwise temporal Conv2d (10 × 1),  F2 → F2 filters

    Output shape: (N, F2, T, 1)
    """

    def __init__(self, F1, D, kern_length, in_chans, dropout=0.1):
        super().__init__()
        F2 = F1 * D

        # Block 1 – temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (kern_length, 1), padding='same', bias=True)
        self.norm1 = nn.BatchNorm2d(F1)
        self.act1  = nn.ELU()

        # Block 2 – depthwise convolution collapses sensor channels
        # groups=F1 makes each of the F1 feature maps act as a depthwise group
        # with depth_multiplier D → output channels = F1*D = F2
        self.dw_conv = nn.Conv2d(F1, F2, (1, in_chans), groups=F1, bias=True)
        self.norm2   = nn.BatchNorm2d(F2)
        self.act2    = nn.ELU()
        self.drop2   = nn.Dropout(dropout)

        # Block 3 – pointwise temporal convolution
        self.conv3 = nn.Conv2d(F2, F2, (10, 1), padding='same', bias=True)
        self.norm3 = nn.BatchNorm2d(F2)
        self.act3  = nn.ELU()
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, 1, T, in_chans)
        x = self.act1(self.norm1(self.conv1(x)))          # (N, F1, T, in_chans)
        x = self.drop2(self.act2(self.norm2(self.dw_conv(x))))  # (N, F2, T, 1)
        x = self.drop3(self.act3(self.norm3(self.conv3(x))))    # (N, F2, T, 1)
        return x


class DIBlock(nn.Module):
    """
    Dilated temporal convolutional block equivalent to DI_block_() in Keras.

    Expects input (N, T, C) — sequence-first; permutes internally for Conv1d.

    Two pairs of causal Conv1d layers:
      pair 1: dilation = 1           with residual projection if needed
      pair 2: dilation = 2^(i+1)     with additive residual from pair 1 output

    Output shape: (N, T, filters)
    """

    def __init__(self, in_channels, filters, kernel_size, dropout,
                 activation='relu', i=0):
        super().__init__()
        act_cls  = nn.ELU if activation == 'elu' else nn.ReLU
        dilation2 = 2 ** (i + 1)

        # --- Pair 1 (dilation = 1) ---
        self.conv1  = CausalConv1d(in_channels, filters, kernel_size, dilation=1)
        self.bn1    = nn.BatchNorm1d(filters)
        self.act1   = act_cls()
        self.drop1  = nn.Dropout(dropout)

        self.conv2  = CausalConv1d(filters, filters, kernel_size, dilation=1)
        self.bn2    = nn.BatchNorm1d(filters)
        self.act2   = act_cls()
        self.drop2  = nn.Dropout(dropout)

        # Residual projection when channel dims differ
        self.res_conv = (
            nn.Conv1d(in_channels, filters, 1)
            if in_channels != filters else None
        )
        self.act_res1 = act_cls()

        # --- Pair 2 (dilation = 2^(i+1)) ---
        self.conv3  = CausalConv1d(filters, filters, kernel_size, dilation=dilation2)
        self.bn3    = nn.BatchNorm1d(filters)
        self.act3   = act_cls()
        self.drop3  = nn.Dropout(dropout)

        self.conv4  = CausalConv1d(filters, filters, kernel_size, dilation=dilation2)
        self.bn4    = nn.BatchNorm1d(filters)
        self.act4   = act_cls()
        self.drop4  = nn.Dropout(dropout)

        self.act_res2 = act_cls()

    def forward(self, x):
        # x: (N, T, C) — permute to (N, C, T) for Conv1d
        x = x.permute(0, 2, 1)          # (N, C, T)

        residual = x

        # Pair 1
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        if self.res_conv is not None:
            residual = self.res_conv(residual)
        out = self.act_res1(out + residual)

        # Pair 2
        out2 = self.drop3(self.act3(self.bn3(self.conv3(out))))
        out2 = self.drop4(self.act4(self.bn4(self.conv4(out2))))
        out  = self.act_res2(out2 + out)

        return out.permute(0, 2, 1)      # (N, T, filters)


# ======================================================================
# Full model
# ======================================================================

class PostFusion(nn.Module):
    """
    PyTorch reimplementation of Post_Fusion() from the original Keras code.

    Input:  (N, 1, in_samples, in_chans)
    Output: (N, n_classes)  — raw logits (no softmax)

    Architecture
    ------------
    1. Slice last sensor channel as "CAP", remaining as "IMU".
    2. Each branch passes through a ConvBlock.
    3. Concatenate along feature dim and squeeze spatial dim → (N, T, 2*F2).
    4. n_windows sliding windows, each processed by:
         AttentionBlock → DIBlock → Dense head
    5. Average the n_windows Dense outputs → logits.
    """

    def __init__(self, n_classes, in_chans=7, in_samples=80, n_windows=4,
                 F1=16, D=2, kernelSize=64, dropout=0.1,
                 di_kernelSize=4, di_filters=32, di_dropout=0.3,
                 di_activation='elu'):
        super().__init__()

        F2       = F1 * D
        embed_dim = 2 * F2          # channels after IMU+CAP concatenation

        self.n_windows  = n_windows
        self.in_samples = in_samples

        # IMU branch: all in_chans channels
        self.imu_block = ConvBlock(F1, D, kernelSize, in_chans=in_chans,
                                   dropout=dropout)
        # CAP branch: last channel only (in_chans=1)
        self.cap_block = ConvBlock(F1, D, kernelSize, in_chans=1,
                                   dropout=dropout)

        # Per-window components (separate weights, matching Keras loop behaviour)
        self.attn_blocks  = nn.ModuleList([
            AttentionBlock(embed_dim) for _ in range(n_windows)
        ])
        self.di_blocks = nn.ModuleList([
            DIBlock(embed_dim, di_filters, di_kernelSize,
                    di_dropout, di_activation, i=0)
            for _ in range(n_windows)
        ])
        self.dense_heads = nn.ModuleList([
            nn.Linear(di_filters, n_classes) for _ in range(n_windows)
        ])

    def forward(self, x):
        # x: (N, 1, T, in_chans)

        # --- Slice IMU and CAP ---
        imu = x                        # (N, 1, T, in_chans)
        cap = x[:, :, :, -1:]          # (N, 1, T, 1)

        # --- Conv branches ---
        imu_feat = self.imu_block(imu)  # (N, F2, T, 1)
        cap_feat = self.cap_block(cap)  # (N, F2, T, 1)

        # --- Concatenate and reshape to sequence format ---
        block1 = torch.cat([imu_feat, cap_feat], dim=1)  # (N, 2*F2, T, 1)
        block1 = block1.squeeze(-1).permute(0, 2, 1)      # (N, T, 2*F2)

        # --- Sliding windows ---
        T          = block1.shape[1]
        window_len = T - self.n_windows + 1

        outputs = []
        for i in range(self.n_windows):
            win = block1[:, i : i + window_len, :]          # (N, window_len, 2*F2)
            win = self.attn_blocks[i](win)                   # (N, window_len, 2*F2)
            win = self.di_blocks[i](win)                     # (N, window_len, di_filters)
            win = win[:, -1, :]                              # (N, di_filters)
            outputs.append(self.dense_heads[i](win))         # (N, n_classes)

        # Average across windows → logits
        out = torch.stack(outputs, dim=0).mean(dim=0)       # (N, n_classes)
        return out


def Post_Fusion(n_classes, in_chans=7, in_samples=80, n_windows=4,
                F1=16, D=2, kernelSize=64, dropout=0.1,
                di_kernelSize=4, di_filters=32, di_dropout=0.3,
                di_activation='elu'):
    """Factory function that returns a PostFusion nn.Module."""
    return PostFusion(
        n_classes, in_chans, in_samples, n_windows, F1, D,
        kernelSize, dropout, di_kernelSize, di_filters,
        di_dropout, di_activation
    )
