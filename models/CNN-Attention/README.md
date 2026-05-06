# CNN-Attention: Gym Exercise Classifier

Hybrid CNN + Multi-Head Attention + Dilated Causal Convolution model for classifying gym exercises from wearable sensor data. The default task is 2-class parent classification (arm curl vs. bench press) using Leave-One-Subject-Out (LOSO) cross-validation.

---

## Architecture

The model (`PostFusion` in [models.py](models.py)) processes windowed sensor data through two parallel CNN branches:

- **IMU branch** — ConvBlock on all input channels (temporal → depthwise → pointwise convolution)
- **CAP branch** — ConvBlock on the last sensor channel only

The branch outputs are concatenated, then passed through `n_windows` sliding windows. Each window is processed by:
1. `AttentionBlock` — multi-head self-attention with pre-norm and residual
2. `DIBlock` — dilated causal convolutional block with two dilation levels
3. Linear head → logits

Final prediction is the average of all window logits.

---

## Environment Setup

```bash
cd /path/to/CNN-Attention
conda create recym
conda activate recgym
pip install -r requirements.txt
```

---

## Configuration

All settings are defined inside the `run()` function in [main_TrainTest.py](main_TrainTest.py). Edit these before running:

```python
# --- Data ---
data_path      = "/path/to/goldendataset.csv"   # training data (golden dataset)
test_data_path = "/path/to/OurData.csv"   # generalization test data

# --- Output ---
results_path = os.path.join(os.getcwd(), "path/to/results")

# --- Dataset config ---
in_samples  = 40    # sliding window size (samples)
n_channels  = 6     # number of sensor channels
n_classes   = 2     # 2 for parent classification; 10 for child-class
classes_labels = ['arm curl', 'bench press']

# --- Training config ---
batch_size  = 32
epochs      = 10
lr          = 0.0001
n_train     = 1     # number of independent runs per LOSO fold
LearnCurves = True  # save learning curve plots
```


## Running Training and Testing

```bash
conda activate recgym
cd /path/to/CNN-Attention
python main_TrainTest.py
```

This calls `run()`, which executes the full LOSO pipeline:

1. **Train** — for each subject, trains on all other subjects (LOSO). Uses Adam optimizer with StepLR scheduler (LR halved every 30 epochs). Best checkpoint per fold is saved by validation accuracy. Mixed precision (AMP) is used automatically when a GPU is available.

2. **Test** — loads the best checkpoint for each fold and evaluates on the held-out subject. Saves per-subject confusion matrices, prediction arrays, and a summary bar chart.

3. **Generalization test** — evaluates the full LOSO ensemble (averaged softmax probabilities) on `test_data_path`. Useful for testing on a dataset not seen during training.

---

## Outputs

All outputs are written to `results_path`:

```
path/to/results
├── saved models/
│   └── run-1/
│       └── subject-{id}.pt          # best model checkpoint per fold
├── best models.txt                  # paths to best checkpoint per subject
├── log.txt                          # training + test metrics log
├── perf_allRuns.npz                 # accuracy and kappa arrays
├── learning_curves_subject_{id}.png # train/val accuracy and loss curves
├── cm_subject_{id}.png              # per-subject confusion matrix
├── bar_Accuracy_per_subject.png
├── bar_Kappa_per_subject.png
└── generalization_test/
    ├── log.txt
    ├── cm_subject_{id}.png
    ├── bar_f1 score_per_subject.png
    ├── bar_kappa_per_subject.png
    └── test/
        ├── Y_truth_Sub_{id}.npy
        └── Y_pred_Sub_{id}.npy
```

---

## Optional Studies

The `run()` function contains commented-out blocks for additional experiments. Uncomment the relevant section to run it.

### Study A — Within-Participant (2-class)
Trains and tests a separate model per participant using only that participant's own data. No cross-subject information is used.

```python
train_test_within_participant(
    dataset_conf, test_data_path, results_wip_path,
    n_train_ac=7, n_train_bp=10,   # training IDs per class
    n_test_ac=1,  n_test_bp=2,     # held-out test IDs per class
    epochs=100, batch_size=32, lr=1e-4,
)
```

### Study B — Progressive Fine-Tuning (child-class)
Averages the LOSO ensemble weights, then progressively fine-tunes on child-class data (exercise variants), evaluating generalization vs. catastrophic forgetting at each step.

```python
fine_tune(
    dataset_conf, test_data_path, results_path, results_ft_path,
    eval_participants=CHILD_EVAL_PARTICIPANTS,
    finetune_epochs=20,
    finetune_lr=1e-5,
    batch_size=32,
)
```

### Study C — Within-Participant (10-class child)
Same as Study A but treats each `(parent, child)` exercise pair as a separate class, training a 10-class model per participant.

```python
train_test_within_participant_childclass(
    dataset_conf, test_data_path, results_wip10_path,
    n_test_ids_per_pair=1,
    epochs=100, batch_size=32, lr=1e-4,
)
```

### STUDY C - Within-Dataset (LOSOCV, 10-class child)
Leave-one-participant-out cross-validation for the 10-class problem.

```python
train_test_loso_10class(
    dataset_conf, test_data_path, results_loocv10_path,
    epochs=100, batch_size=32, lr=1e-4,
)
```

---

## File Overview

| File | Description |
|------|-------------|
| [main_TrainTest.py](main_TrainTest.py) | Entry point — `run()`, training loop, evaluation, fine-tuning |
| [models.py](models.py) | `PostFusion` model, `ConvBlock`, `DIBlock`, `CausalConv1d` |
| [attention_models.py](attention_models.py) | `AttentionBlock` (multi-head self-attention) |
| [preprocess_golden_v3.py](preprocess_golden_v3.py) | Data loading, windowing, normalization |
| [requirements.txt](requirements.txt) | Python dependencies |
