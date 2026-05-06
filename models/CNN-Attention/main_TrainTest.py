import os
import gc
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.utils import class_weight, shuffle as sk_shuffle
from sklearn.metrics import classification_report

import models
from preprocess_golden_v3 import (get_data, get_unique_subjects, load_new_dataset,
                                   load_child_class_data, load_golden_for_forgetting,
                                   load_ourdata_all,
                                   CHILD_EVAL_PARTICIPANTS, CHILD_FINETUNE_PARTICIPANTS,
                                   GOLDEN_CHILD_TO_PARENT, GOLDEN_CHILD_LABELS,
                                   GOLDEN_PARENT_LABELS,
                                   OURDATA_PAIR_TO_CLASS, OURDATA_10CLASS_LABELS,
                                   OURDATA_10CLASS_TO_PARENT)

print(matplotlib.get_backend())


# ======================================================================
# Helpers
# ======================================================================

class History:
    """Lightweight Keras-like history object for learning-curve plots."""
    def __init__(self):
        self.history = {
            'accuracy': [], 'val_accuracy': [],
            'loss':     [], 'val_loss':     []
        }


def draw_learning_curves(history, results_path, sub):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title(f'Subject {sub} – Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc='upper left')

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title(f'Subject {sub} – Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(results_path, f"learning_curves_subject_{sub}.png"))
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, sub, results_path, classes,
                          normalize=True, cmap=plt.cm.Blues):
    accuracy = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, average='micro')
    title    = (f"Sub: {sub}  Macro F1: {round(f1*100, 2)}"
                f"  Accuracy: {round(accuracy*100, 2)}")

    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt    = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(results_path, f"cm_subject_{sub}.png"))
    plt.close(fig)
    return ax


def draw_performance_barChart(subjects, metric, label, results_path):
    fig, ax = plt.subplots()
    x = list(range(len(subjects)))
    bars = ax.bar(x, metric, 0.5, label=label)
    for bar, val in zip(bars, metric):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    mean_val = np.mean(metric)
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=1,
               label=f'Mean: {mean_val:.3f}')
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects], rotation=45)
    ax.set_title(f'Model {label} per subject (LOSO)')
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_path, f"bar_{label.lower()}_per_subject.png"))
    plt.close(fig)


def plot_child_class_bar(child_labels, child_accs, parent_labels, child_to_parent,
                         results_path, filename, title):
    """
    Bar chart of per-child-class parent accuracy, grouped and coloured by parent.

    Parameters
    ----------
    child_labels    : list[str]  – display label for each child class index
    child_accs      : dict       – {child_idx: accuracy}  (missing = not plotted)
    parent_labels   : list[str]  – display label for each parent class index
    child_to_parent : dict       – child_idx → parent_idx
    results_path    : str
    filename        : str        – output filename (no directory)
    title           : str        – plot title
    """
    # Build ordered lists (child classes sorted by parent, then by child index)
    ordered_indices = []
    for p_idx in range(len(parent_labels)):
        ordered_indices.extend(
            sorted([c for c, p in child_to_parent.items() if p == p_idx])
        )

    # Filter to those present in child_accs
    present = [c for c in ordered_indices if c in child_accs]
    if not present:
        return

    labels  = [child_labels[c] for c in present]
    accs    = [child_accs[c]   for c in present]
    colors  = [f'C{child_to_parent[c]}' for c in present]

    fig, ax = plt.subplots(figsize=(max(6, len(present) * 0.9), 4))
    bars = ax.bar(range(len(present)), accs, color=colors, width=0.6, edgecolor='white')

    # Accuracy value on top of each bar
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Parent Class Accuracy')
    ax.set_ylim([0, 1.1])
    ax.set_title(title)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='chance')

    # Legend for parent groups
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=f'C{p}', label=parent_labels[p])
                      for p in range(len(parent_labels))]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(results_path, filename))
    plt.close(fig)


# ======================================================================
# PyTorch inference helper
# ======================================================================

def predict_batched(model, X_np, device, batch_size=256):
    """Run inference in mini-batches; returns numpy integer predictions."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_np), batch_size):
            batch = torch.tensor(X_np[start:start + batch_size]).to(device)
            out   = model(batch)
            preds.append(out.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


# ======================================================================
# Training  (LOSO)
# ======================================================================

def train(dataset_conf, train_conf, results_path):
    in_exp = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    best_models_file = open(os.path.join(results_path, "best models.txt"), "w")
    log_write        = open(os.path.join(results_path, "log.txt"),         "w")
    perf_allRuns     = open(os.path.join(results_path, "perf_allRuns.npz"), 'wb')

    dataset   = dataset_conf.get('name')
    data_path = dataset_conf.get('data_path')
    subjects  = dataset_conf.get('subjects')
    n_sub     = len(subjects)

    batch_size  = train_conf.get('batch_size')
    epochs      = train_conf.get('epochs')
    lr          = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    n_train     = train_conf.get('n_train')

    acc   = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    for s_idx, test_sub in enumerate(subjects):
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        in_sub = time.time()
        print(f'\n{"="*60}')
        print(f'LOSO fold {s_idx+1}/{n_sub}  –  Test subject: {test_sub}')
        print(f'{"="*60}')
        log_write.write(f'\nLOSO fold {s_idx+1}/{n_sub}  –  Test subject: {test_sub}\n')

        BestSubjAcc        = 0
        bestTrainingHistory = None

        # ---- Load data ------------------------------------------------- #
        X_train, y_train_labels, y_train_onehot, \
        X_test,  y_test_labels,  y_test_onehot = get_data(
            data_path, dataset, test_subject=test_sub, shuffle_train=False
        )
        print("\n--- DATA DIAGNOSTICS ---")
        print(f"X_train shape:        {X_train.shape}")
        print(f"X_train NaN count:    {np.isnan(X_train).sum()}")
        print(f"X_train Inf count:    {np.isinf(X_train).sum()}")
        print(f"X_train min/max:      {X_train.min():.4f} / {X_train.max():.4f}")
        print(f"X_train mean/std:     {X_train.mean():.4f} / {X_train.std():.4f}")
        print(f"y_train unique:       {np.unique(y_train_labels)}")
        print(f"X_test NaN count:     {np.isnan(X_test).sum()}")

        # ---- Class weights --------------------------------------------- #
        class_weights_arr = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        class_weight_dict = dict(enumerate(class_weights_arr))
        print(f"  Class weights: {class_weight_dict}")

        del y_train_onehot
        gc.collect()

        val_split_ratio = 0.2
        class_weights_t = torch.tensor(class_weights_arr, dtype=torch.float32).to(device)

        # ---- Multiple runs per fold ------------------------------------ #
        for run in range(n_train):
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            torch.manual_seed(run + 1)
            np.random.seed(run + 1)

            # Re-shuffle and re-split per run so each run sees a different val set
            X_tr_shuf, y_tr_shuf = sk_shuffle(X_train, y_train_labels, random_state=run + 1)
            val_size      = int(len(X_tr_shuf) * val_split_ratio)
            X_train_split = X_tr_shuf[:-val_size]
            X_val_split   = X_tr_shuf[-val_size:]
            y_train_split = y_tr_shuf[:-val_size].astype(np.int64)
            y_val_split   = y_tr_shuf[-val_size:].astype(np.int64)

            # Build tensors / DataLoader
            X_train_t = torch.tensor(X_train_split)
            y_train_t = torch.tensor(y_train_split)
            X_val_t   = torch.tensor(X_val_split)
            y_val_t   = torch.tensor(y_val_split)

            train_ds = TensorDataset(X_train_t, y_train_t)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=torch.cuda.is_available())

            in_run  = time.time()
            run_dir = os.path.join(results_path, 'saved models', f'run-{run+1}')
            os.makedirs(run_dir, exist_ok=True)
            filepath = os.path.join(run_dir, f'subject-{test_sub}.pt')

            model     = getModel(dataset_conf).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_t)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # Halve LR at epochs 30, 60, 90 — matching the original scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.5
            )
            use_amp  = torch.cuda.is_available()
            scaler   = GradScaler(enabled=use_amp)

            history     = History()
            best_val_acc = 0.0

            for epoch in range(epochs):
                # --- Train ---
                model.train()
                epoch_loss = 0.0
                correct    = 0
                total      = 0

                for X_batch, y_batch in train_dl:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    with autocast(enabled=use_amp):
                        logits = model(X_batch)
                        loss   = criterion(logits, y_batch)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item() * len(y_batch)
                    correct    += (logits.argmax(-1) == y_batch).sum().item()
                    total      += len(y_batch)

                train_acc  = correct / total
                train_loss = epoch_loss / total

                # --- Validate ---
                model.eval()
                with torch.no_grad():
                    X_val_d = X_val_t.to(device)
                    y_val_d = y_val_t.to(device)
                    with autocast(enabled=use_amp):
                        val_logits = model(X_val_d)
                        val_loss   = criterion(val_logits, y_val_d).item()
                    val_pred = val_logits.argmax(-1)
                    val_acc  = (val_pred == y_val_d).float().mean().item()

                scheduler.step()

                history.history['accuracy'].append(train_acc)
                history.history['val_accuracy'].append(val_acc)
                history.history['loss'].append(train_loss)
                history.history['val_loss'].append(val_loss)

                print(f"  Epoch {epoch+1:3d}/{epochs}"
                      f"  loss: {train_loss:.4f}  acc: {train_acc:.4f}"
                      f"  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), filepath)

            # Load best weights and evaluate on test set
            model.load_state_dict(torch.load(filepath, map_location=device))
            y_pred = predict_batched(model, X_test, device, batch_size=256)
            labels = y_test_labels

            acc[s_idx, run]   = accuracy_score(labels, y_pred)
            kappa[s_idx, run] = cohen_kappa_score(labels, y_pred)

            out_run = time.time()
            info = (f'  Subject: {test_sub}   Run: {run+1}'
                    f'   Time: {(out_run-in_run)/60:.1f} m'
                    f'   Test acc: {acc[s_idx, run]:.4f}'
                    f'   Test kappa: {kappa[s_idx, run]:.4f}')
            print(info)
            log_write.write(info + '\n')

            if BestSubjAcc < acc[s_idx, run]:
                BestSubjAcc        = acc[s_idx, run]
                bestTrainingHistory = history

            del model, y_pred, labels
            del X_train_split, X_val_split, y_train_split, y_val_split
            del X_train_t, y_train_t, X_val_t, y_val_t, train_ds, train_dl
            gc.collect()

        # ---- Best run summary ----------------------------------------- #
        best_run  = int(np.argmax(acc[s_idx, :]))
        best_path = f'saved models/run-{best_run+1}/subject-{test_sub}.pt\n'
        best_models_file.write(best_path)

        out_sub = time.time()
        info = (f'\n  Subject {test_sub} summary'
                f'   best_run: {best_run+1}'
                f'   Time: {(out_sub-in_sub)/60:.1f} m\n'
                f'  best_acc: {acc[s_idx, best_run]:.4f}'
                f'   avg_acc: {np.mean(acc[s_idx,:]):.4f} ± {acc[s_idx,:].std():.4f}\n'
                f'  best_kappa: {kappa[s_idx, best_run]:.4f}'
                f'   avg_kappa: {np.mean(kappa[s_idx,:]):.4f} ± {kappa[s_idx,:].std():.4f}')
        print(info)
        log_write.write(info + '\n')

        if LearnCurves and bestTrainingHistory is not None:
            draw_learning_curves(bestTrainingHistory, results_path, test_sub)

        del X_train, y_train_labels
        del X_test, y_test_labels, y_test_onehot
        del bestTrainingHistory
        gc.collect()

    # ---- Overall summary ---------------------------------------------- #
    out_exp = time.time()
    info = f'\nTotal training time: {(out_exp-in_exp)/3600:.1f} h\n'
    print(info)
    log_write.write(info)

    np.savez(perf_allRuns, acc=acc, kappa=kappa, subjects=subjects)
    best_models_file.close()
    log_write.close()
    perf_allRuns.close()


# ======================================================================
# Evaluation  (LOSO)
# ======================================================================

def test(model, dataset_conf, results_path):
    os.makedirs(os.path.join(results_path, "test"), exist_ok=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_write = open(os.path.join(results_path, "log.txt"), "a")

    with open(os.path.join(results_path, "best models.txt"), "r") as f:
        best_model_paths = [line.strip() for line in f.readlines()]

    dataset        = dataset_conf.get('name')
    data_path      = dataset_conf.get('data_path')
    subjects       = dataset_conf.get('subjects')
    n_sub          = len(subjects)
    classes_labels = dataset_conf.get('cl_labels')

    model = model.to(device)

    acc_bestRun   = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)
    labels_all    = np.array([])
    predic_all    = np.array([])

    for s_idx, test_sub in enumerate(subjects):
        _, _, _, X_test, y_test_labels, y_test_onehot = get_data(
            data_path, dataset, test_subject=test_sub
        )

        filepath = os.path.join(results_path, best_model_paths[s_idx])
        print(f"Loading weights: {filepath}")
        model.load_state_dict(torch.load(filepath, map_location=device))

        y_pred = predict_batched(model, X_test, device, batch_size=256)
        labels = y_test_labels

        acc_bestRun[s_idx]   = accuracy_score(labels, y_pred)
        kappa_bestRun[s_idx] = cohen_kappa_score(labels, y_pred)

        plot_confusion_matrix(labels, y_pred, test_sub, results_path, classes_labels)

        np.save(os.path.join(results_path, "test", f"Y_truth_Sub_{test_sub}.npy"), labels)
        np.save(os.path.join(results_path, "test", f"Y_pred_Sub_{test_sub}.npy"),  y_pred)

        labels_all = np.concatenate((labels_all, labels))
        predic_all = np.concatenate((predic_all, y_pred))

        info = (f'Subject: {test_sub}'
                f'   acc: {acc_bestRun[s_idx]:.4f}'
                f'   kappa: {kappa_bestRun[s_idx]:.4f}')
        print(info)
        log_write.write('\n' + info)

        del X_test, y_test_labels, y_test_onehot, y_pred, labels
        gc.collect()

    info = (f'\nLOSO results across {n_sub} subjects:\n'
            f'Accuracy = {np.mean(acc_bestRun):.4f} ± {acc_bestRun.std():.4f}\n'
            f'Kappa    = {np.mean(kappa_bestRun):.4f} ± {kappa_bestRun.std():.4f}\n')
    print(info)
    log_write.write(info)

    draw_performance_barChart(subjects, acc_bestRun,   'Accuracy', results_path)
    draw_performance_barChart(subjects, kappa_bestRun, 'Kappa',    results_path)
    plot_confusion_matrix(labels_all, predic_all, "All", results_path, list(range(len(classes_labels))))

    print(classification_report(labels_all, predic_all, target_names=classes_labels))
    log_write.close()

def test_general(dataset_conf, gen_data_path, results_path, results_gen_path):
    """
    Evaluate the LOSO ensemble on an entirely new dataset without fine-tuning.

    All N models saved during LOSO training vote on each window via averaged
    softmax probabilities. The best-model-per-subject paths are read from
    results_path/best models.txt.

    Parameters
    ----------
    dataset_conf    : dict  – same conf used during training (architecture params)
    gen_data_path   : str   – path to the new dataset CSV
    results_path    : str   – directory containing the trained LOSO artefacts
    results_gen_path: str   – directory where new evaluation outputs are saved
    """
    os.makedirs(os.path.join(results_gen_path, "test"), exist_ok=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_write = open(os.path.join(results_gen_path, "log.txt"), "w")

    classes_labels = dataset_conf.get('cl_labels')
    dataset        = dataset_conf.get('name')
    n_classes      = dataset_conf.get('n_classes')
    window_size    = dataset_conf.get('in_samples')

    # ---- Load all ensemble model paths --------------------------------- #
    with open(os.path.join(results_path, "best models.txt"), "r") as f:
        best_model_paths = [line.strip() for line in f.readlines()]

    print(f"\nLoading {len(best_model_paths)} ensemble models from {results_path}...")
    ensemble_models = []
    for path in best_model_paths:
        m = getModel(dataset_conf).to(device)
        m.load_state_dict(torch.load(os.path.join(results_path, path), map_location=device))
        m.eval()
        ensemble_models.append(m)
    print(f"Ensemble size: {len(ensemble_models)} models")

    # ---- Load and preprocess the new dataset --------------------------- #
    print(f"\nLoading new dataset from: {gen_data_path}")
    X_all, y_all, subject_ids, y_child_all = load_new_dataset(
        gen_data_path, dataset, window_size=window_size
    )
    subjects = sorted(np.unique(subject_ids).tolist())
    n_sub    = len(subjects)

    # ---- Ensemble inference helper ------------------------------------- #
    softmax_fn = nn.Softmax(dim=-1)

    def ensemble_predict(X_np, batch_size=256):
        """Average softmax probabilities across all models, return argmax."""
        n        = len(X_np)
        avg_prob = np.zeros((n, n_classes), dtype=np.float32)
        for m in ensemble_models:
            with torch.no_grad():
                probs_m = []
                for start in range(0, n, batch_size):
                    batch  = torch.tensor(X_np[start:start + batch_size]).to(device)
                    logits = m(batch)
                    probs_m.append(softmax_fn(logits).cpu().numpy())
                avg_prob += np.concatenate(probs_m, axis=0)
        avg_prob /= len(ensemble_models)
        return avg_prob.argmax(axis=1)

    # ---- Per-subject evaluation ---------------------------------------- #
    f1_per_sub    = np.zeros(n_sub)
    kappa_per_sub = np.zeros(n_sub)
    labels_all    = np.array([])
    predic_all    = np.array([])
    y_child_coll  = np.array([], dtype=int)   # child class per window, same order

    for s_idx, sub in enumerate(subjects):
        mask   = subject_ids == sub
        X_sub  = X_all[mask]
        y_sub  = y_all[mask]
        y_pred = ensemble_predict(X_sub)

        f1_per_sub[s_idx]    = f1_score(y_sub, y_pred,average='micro')
        kappa_per_sub[s_idx] = cohen_kappa_score(y_sub, y_pred)

        plot_confusion_matrix(y_sub, y_pred, sub, results_gen_path, classes_labels)
        np.save(os.path.join(results_gen_path, "test", f"Y_truth_Sub_{sub}.npy"), y_sub)
        np.save(os.path.join(results_gen_path, "test", f"Y_pred_Sub_{sub}.npy"),  y_pred)

        labels_all   = np.concatenate((labels_all,   y_sub))
        predic_all   = np.concatenate((predic_all,   y_pred))
        if y_child_all is not None:
            y_child_coll = np.concatenate((y_child_coll, y_child_all[mask]))

        info = (f'Subject: {sub}'
                f'   acc: {f1_per_sub[s_idx]:.4f}'
                f'   kappa: {kappa_per_sub[s_idx]:.4f}')
        print(info)
        log_write.write(info + '\n')

    # ---- Overall summary ----------------------------------------------- #
    overall_acc = float(accuracy_score(labels_all, predic_all))
    overall_f1  = float(f1_score(labels_all, predic_all, average='macro', zero_division=0))
    per_class_f1 = f1_score(labels_all, predic_all, average=None, zero_division=0)
    ac_mask_all  = labels_all == 0
    bp_mask_all  = labels_all == 1
    ac_acc_all   = float(accuracy_score(labels_all[ac_mask_all], predic_all[ac_mask_all])) if ac_mask_all.any() else 0.0
    bp_acc_all   = float(accuracy_score(labels_all[bp_mask_all], predic_all[bp_mask_all])) if bp_mask_all.any() else 0.0
    ac_f1_all    = float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0
    bp_f1_all    = float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0

    info = (f'\nEnsemble results on new dataset ({n_sub} subjects):\n'
            f'  Arm Curl    — acc: {ac_acc_all:.4f}  f1: {ac_f1_all:.4f}\n'
            f'  Bench Press — acc: {bp_acc_all:.4f}  f1: {bp_f1_all:.4f}\n'
            f'  Overall     — acc: {overall_acc:.4f}  f1: {overall_f1:.4f}\n'
            f'  Kappa       = {np.mean(kappa_per_sub):.4f} ± {kappa_per_sub.std():.4f}\n')
    print(info)
    log_write.write(info)

    draw_performance_barChart(subjects, f1_per_sub,   'f1 score', results_gen_path)
    draw_performance_barChart(subjects, kappa_per_sub, 'kappa',    results_gen_path)
    plot_confusion_matrix(labels_all, predic_all, "All", results_gen_path, classes_labels)

    print(classification_report(labels_all, predic_all, target_names=classes_labels))

    # ---- Per-child-class accuracy + F1 ---------------------------------- #
    if len(y_child_coll) > 0:
        child_header = (f"\n{'='*60}\n"
                        f"Per-child-class accuracy and F1 (zero-shot)\n"
                        f"{'='*60}\n"
                        f"  (performance on windows routed to the correct parent class)\n")
        print(child_header)
        log_write.write(child_header)

        child_accs = {}
        for p_idx, p_label in enumerate(GOLDEN_PARENT_LABELS):
            group_line = f"\n  [{p_label}]\n"
            print(group_line, end="")
            log_write.write(group_line)
            child_classes = sorted([c for c, p in GOLDEN_CHILD_TO_PARENT.items()
                                    if p == p_idx])
            for c_idx in child_classes:
                cmask = y_child_coll == c_idx
                if cmask.sum() == 0:
                    continue
                y_true_c = np.full(int(cmask.sum()), p_idx, dtype=int)
                y_pred_c = predic_all[cmask].astype(int)
                child_acc = float((y_pred_c == p_idx).mean())
                child_f1  = float(f1_score(y_true_c, y_pred_c,
                                           pos_label=p_idx, average='binary',
                                           zero_division=0))
                child_accs[c_idx] = child_acc
                line = (f"    {GOLDEN_CHILD_LABELS[c_idx]}"
                        f"  acc={child_acc:.4f}"
                        f"  f1={child_f1:.4f}"
                        f"  n={int(cmask.sum())}\n")
                print(line, end="")
                log_write.write(line)

        plot_child_class_bar(
            GOLDEN_CHILD_LABELS, child_accs,
            GOLDEN_PARENT_LABELS, GOLDEN_CHILD_TO_PARENT,
            results_gen_path,
            "bar_child_class_accuracy_baseline.png",
            "Per-child-class parent accuracy (zero-shot)"
        )

    log_write.close()

    for m in ensemble_models:
        del m
    gc.collect()


# ======================================================================
# Model factory
# ======================================================================

def getModel(dataset_conf):
    model = models.Post_Fusion(
        n_classes     = dataset_conf.get('n_classes'),
        in_chans      = dataset_conf.get('n_channels'),
        in_samples    = dataset_conf.get('in_samples'),
        n_windows     = 2,
        F1            = 8,
        D             = 2,
        kernelSize    = 20,
        dropout       = 0.1,
        di_kernelSize = 3,
        di_filters    = 16,
        di_dropout    = 0.4,
        di_activation = 'elu'
    )
    return model


# ======================================================================
# Fine-tuning helpers
# ======================================================================

def _evaluate_childclasses(model, X, y, pairs, device):
    """
    Run inference and return overall accuracy/F1 plus per-(parent,child) accuracy.

    The model outputs parent-class labels (0=arm curl, 1=bench press).
    Child-class performance is measured per (parent,child) pair: for each pair,
    what fraction of windows does the model assign the correct *parent* label?

    Returns
    -------
    dict with keys:
        overall_acc  : float
        overall_f1   : float
        preds        : np.ndarray  – predicted parent labels
        per_pair     : dict  (parent,child) → {accuracy, n_windows, expected}
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            batch = torch.tensor(X[start:start + 256]).to(device)
            preds.append(model(batch).argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)

    overall_acc = accuracy_score(y, preds)
    overall_f1  = f1_score(y, preds,average='micro')

    per_pair = {}
    for p, c in sorted(set(map(tuple, pairs.tolist()))):
        mask     = (pairs[:, 0] == p) & (pairs[:, 1] == c)
        expected = int(y[mask][0])   # all windows in a pair share the same parent label
        acc      = float((preds[mask] == expected).mean())
        per_pair[(p, c)] = {
            'accuracy':  acc,
            'n_windows': int(mask.sum()),
            'expected':  'arm curl' if expected == 0 else 'bench press',
        }

    return {'overall_acc': overall_acc, 'overall_f1': overall_f1,
            'preds': preds, 'per_pair': per_pair}


def plot_finetuning_curve(ft_curve, baseline_childclass_acc, baseline_golden_acc,
                          results_ft_path):
    """
    Progressive accuracy curve matching the notebook (cell 52).

    X-axis  : number of IDs sampled per child-class pair
    Y-axis  : accuracy (%)
    Blue    : child-class eval accuracy
    Green   : golden dataset accuracy  (catastrophic-forgetting monitor)
    Dotted  : pre-fine-tuning baselines
    Secondary x-axis: equivalent % of fine-tune IDs used
    """
    n_ids_vals  = [r['n_ids_per_pair'] for r in ft_curve]
    pct_vals    = [r['pct']            for r in ft_curve]
    child_accs  = [r['childclass_acc'] * 100 for r in ft_curve]
    golden_accs = [r['golden_acc']     * 100 for r in ft_curve]

    fig, ax = plt.subplots(figsize=(max(8, len(n_ids_vals) * 1.5), 5))

    ax.plot(n_ids_vals, child_accs, 'o-', color='#3498db', linewidth=2,
            markersize=8, label='Child-class accuracy (eval participant)')
    ax.plot(n_ids_vals, golden_accs, 's--', color='#2ecc71', linewidth=2,
            markersize=8, label='Golden accuracy (catastrophic-forgetting monitor)')

    ax.axhline(baseline_childclass_acc * 100, color='#3498db', linestyle=':',
               alpha=0.5,
               label=f'Baseline child-class: {baseline_childclass_acc*100:.1f}%')
    ax.axhline(baseline_golden_acc * 100, color='#2ecc71', linestyle=':',
               alpha=0.5,
               label=f'Baseline golden: {baseline_golden_acc*100:.1f}%')

    for x, y_val in zip(n_ids_vals, child_accs):
        ax.annotate(f'{y_val:.1f}%', (x, y_val),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9, color='#3498db')

    ax.set_xlabel('Number of IDs per Child-Class Pair (used for fine-tuning)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Progressive Fine-Tuning: Child-Class Generalization vs Core Retention')
    ax.set_xticks(n_ids_vals)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(n_ids_vals)
    ax2.set_xticklabels([f'{p:.0f}%' for p in pct_vals])
    ax2.set_xlabel('Equivalent % of Fine-Tune IDs')

    fig.tight_layout()
    fig.savefig(os.path.join(results_ft_path, 'finetuning_curve.png'), dpi=150)
    plt.close(fig)
    print('  Saved: finetuning_curve.png')


def plot_perpair_heatmap(ft_curve, baseline_per_pair, child_pairs, results_ft_path):
    """
    Heatmap of per-(parent,child) accuracy at each fine-tuning step.

    Rows    : child-class pairs  e.g. (1,1), (1,2), …
    Columns : Baseline, 1 ID/pair, 2 IDs/pair, …
    Color   : RdYlGn  0–100%
    """
    step_labels = ['Baseline'] + [
        f"{r['n_ids_per_pair']} ID{'s' if r['n_ids_per_pair'] != 1 else ''}/pair"
        for r in ft_curve
    ]
    pair_labels = [f'({p},{c})' for p, c in child_pairs]
    n_pairs     = len(child_pairs)
    n_steps     = len(step_labels)

    heat = np.zeros((n_pairs, n_steps), dtype=float)
    for i, pair in enumerate(child_pairs):
        heat[i, 0] = baseline_per_pair.get(pair, {}).get('accuracy', 0.0) * 100
        for j, r in enumerate(ft_curve):
            heat[i, j + 1] = r['per_pair'].get(pair, {}).get('accuracy', 0.0) * 100

    fig, ax = plt.subplots(figsize=(max(6, n_steps * 1.4), max(4, n_pairs * 0.8)))
    im = ax.imshow(heat, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(step_labels, rotation=45, ha='right')
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_labels)
    ax.set_title('Per-Pair Classification Accuracy (%): Baseline → Progressive Fine-Tuning')

    for i in range(n_pairs):
        for j in range(n_steps):
            val   = heat[i, j]
            color = 'white' if val < 40 or val > 85 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=9, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Accuracy (%)')
    fig.tight_layout()
    fig.savefig(os.path.join(results_ft_path, 'finetuning_heatmap.png'), dpi=150)
    plt.close(fig)
    print('  Saved: finetuning_heatmap.png')


def _unused_evaluate_child_parent(y_true_child, y_pred_child, child_to_parent,
                           child_labels, parent_labels, results_path, tag, log_write):
    """
    Compute and log accuracy/kappa/classification-report at both child-class
    and parent-class levels. Saves confusion matrices for both levels.
    """
    from sklearn.metrics import classification_report as cls_report

    # ---- Child-class metrics ------------------------------------------- #
    child_acc   = accuracy_score(y_true_child, y_pred_child)
    child_kappa = cohen_kappa_score(y_true_child, y_pred_child,
                                    labels=list(range(len(child_labels))))
    child_f1    = f1_score(y_true_child, y_pred_child, average='micro',
                           labels=list(range(len(child_labels))), zero_division=0)

    # ---- Parent-class metrics (map child → parent) ---------------------- #
    y_true_parent = np.array([child_to_parent[c] for c in y_true_child])
    y_pred_parent = np.array([child_to_parent[c] for c in y_pred_child])
    parent_acc   = accuracy_score(y_true_parent, y_pred_parent)
    parent_kappa = cohen_kappa_score(y_true_parent, y_pred_parent,
                                     labels=list(range(len(parent_labels))))
    parent_f1    = f1_score(y_true_parent, y_pred_parent, average='micro',
                            labels=list(range(len(parent_labels))), zero_division=0)

    # ---- Confusion matrices -------------------------------------------- #
    plot_confusion_matrix(y_true_child,  y_pred_child,
                          f"{tag}_child",  results_path, child_labels)
    plot_confusion_matrix(y_true_parent, y_pred_parent,
                          f"{tag}_parent", results_path, parent_labels)

    # ---- Logging -------------------------------------------------------- #
    header = f"\n{'='*60}\n[{tag}] Evaluation results\n{'='*60}"
    child_info = (f"\n  -- Child-class level --\n"
                  f"  Accuracy  : {child_acc:.4f}\n"
                  f"  Macro F1  : {child_f1:.4f}\n"
                  f"  Kappa     : {child_kappa:.4f}\n")
    parent_info = (f"\n  -- Parent-class level --\n"
                   f"  Accuracy  : {parent_acc:.4f}\n"
                   f"  Macro F1  : {parent_f1:.4f}\n"
                   f"  Kappa     : {parent_kappa:.4f}\n")
    child_report  = "\n  [Child classes]\n"  + cls_report(
        y_true_child,  y_pred_child,  target_names=child_labels,  zero_division=0)
    parent_report = "\n  [Parent classes]\n" + cls_report(
        y_true_parent, y_pred_parent, target_names=parent_labels, zero_division=0)

    # ---- Per-child-class parent accuracy (notebook approach) -------------- #
    # For each child class variant, report what fraction of predictions go to
    # the correct parent class.  This directly matches evaluate_childclasses()
    # in the notebook:  accuracy = (preds == expected_parent).mean() per pair.
    breakdown_header = (
        "\n  -- Per-child-class parent accuracy --\n"
        "  (fraction of windows correctly routed to the right parent class)"
    )
    breakdown_lines = [breakdown_header]
    child_accs = {}
    for p_idx, p_label in enumerate(parent_labels):
        breakdown_lines.append(f"\n  [{p_label}]")
        child_classes = sorted([c for c, p in child_to_parent.items() if p == p_idx])
        for c_idx in child_classes:
            mask = y_true_child == c_idx
            if mask.sum() == 0:
                continue
            parent_acc_c = (y_pred_parent[mask] == p_idx).mean()
            child_accs[c_idx] = parent_acc_c
            breakdown_lines.append(
                f"    {child_labels[c_idx]}"
                f"  parent_acc={parent_acc_c:.4f}"
                f"  n={int(mask.sum())}"
            )

    for line in breakdown_lines:
        print(line)
        log_write.write(line + '\n')

    plot_child_class_bar(
        child_labels, child_accs,
        parent_labels, child_to_parent,
        results_path,
        f"bar_child_class_accuracy_{tag}.png",
        f"Per-child-class parent accuracy [{tag}]"
    )

    for line in [header, child_info, parent_info, child_report, parent_report]:
        print(line)
        log_write.write(line + '\n')

    return child_acc, child_kappa, parent_acc, parent_kappa


def fine_tune(dataset_conf, test_data_path, results_path, results_ft_path,
              eval_participants=None, finetune_epochs=20, finetune_lr=None,
              batch_size=32, random_seed=42):
    """
    Progressive fine-tuning of the pretrained LOSO ensemble on child-class data.

    Strategy
    --------
    1. Load child-class data from test_data_path; split by participant:
         - eval_participants (default: [15])   → held-out evaluation set
         - remaining participants               → fine-tuning pool
    2. Average all LOSO model weights into a single starting model.
    3. Baseline: evaluate the averaged model on both the child-class eval set
       and the original golden dataset (before any fine-tuning).
    4. Progressive loop — for n_ids = 1, 2, …, max_ids_per_pair:
         a. Sample exactly n_ids IDs from each (parent,child) pair in the
            fine-tune pool, stratified per pair (independent each step).
         b. Reload clean averaged weights (each step is independent).
         c. Fine-tune ALL model parameters with a lower learning rate
            (finetune_lr, default = LOSO lr / 10) — preserves learned features.
         d. Evaluate child-class accuracy on the eval set per pair + overall.
         e. Evaluate golden dataset accuracy (catastrophic-forgetting monitor).
    5. Save best fine-tuned model; plot accuracy curve + per-pair heatmap.

    Parameters
    ----------
    dataset_conf      : dict  – same conf used during LOSO training
    test_data_path    : str   – path to child-class CSV (e.g. OurData_v5.csv)
    results_path      : str   – directory containing LOSO artefacts + "best models.txt"
    results_ft_path   : str   – output directory for fine-tuning results
    eval_participants : list  – participant IDs held out for evaluation (default [15])
    finetune_epochs   : int   – training epochs per progressive step
    finetune_lr       : float – learning rate; defaults to 1e-4 / 10 = 1e-5
    batch_size        : int
    random_seed       : int
    """
    if eval_participants is None:
        eval_participants = CHILD_EVAL_PARTICIPANTS

    os.makedirs(results_ft_path, exist_ok=True)
    os.makedirs(os.path.join(results_ft_path, 'test'), exist_ok=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_write = open(os.path.join(results_ft_path, 'log_finetune.txt'), 'w')

    dataset     = dataset_conf.get('name')
    window_size = dataset_conf.get('in_samples')
    if finetune_lr is None:
        finetune_lr = dataset_conf.get('lr', 1e-4) / 10

    # ------------------------------------------------------------------ #
    # 1. Load child-class data and split by participant
    # ------------------------------------------------------------------ #
    X_all, y_all, pairs_all, ids_all, parts_all, child_pairs, parent_label_map = \
        load_child_class_data(test_data_path, dataset, window_size=window_size)

    eval_mask = np.isin(parts_all, eval_participants)
    ft_mask   = ~eval_mask

    X_eval,  y_eval  = X_all[eval_mask],  y_all[eval_mask]
    pairs_eval        = pairs_all[eval_mask]

    X_ft,  y_ft      = X_all[ft_mask],    y_all[ft_mask]
    pairs_ft          = pairs_all[ft_mask]
    ids_ft            = ids_all[ft_mask]

    ft_parts = sorted(set(parts_all[ft_mask].tolist()))

    info = (f"Eval participants  : {eval_participants}\n"
            f"Fine-tune parts.   : {ft_parts}\n"
            f"Child-class pairs  : {child_pairs}\n"
            f"Fine-tune windows  : {len(X_ft)}\n"
            f"Eval windows       : {len(X_eval)}\n"
            f"Epochs/step        : {finetune_epochs}\n"
            f"Learning rate      : {finetune_lr:.2e}\n")
    print(info);  log_write.write(info + '\n')

    # ------------------------------------------------------------------ #
    # 2. Count IDs per pair in fine-tune pool
    # ------------------------------------------------------------------ #
    pair_id_counts = {}
    for pair in child_pairs:
        p, c = pair
        pmask = (pairs_ft[:, 0] == p) & (pairs_ft[:, 1] == c)
        pair_id_counts[pair] = len(np.unique(ids_ft[pmask]))

    max_ids_per_pair = max(pair_id_counts.values())
    total_ft_ids     = sum(pair_id_counts.values())

    id_info = 'Fine-tune IDs per pair:\n'
    for pair, n in sorted(pair_id_counts.items()):
        id_info += f'  {pair}: {n} IDs\n'
    id_info += f'Max IDs per pair: {max_ids_per_pair}  |  Total: {total_ft_ids}\n'
    print(id_info);  log_write.write(id_info + '\n')

    # ------------------------------------------------------------------ #
    # 3. Build averaged starting model from all LOSO ensemble weights
    # ------------------------------------------------------------------ #
    with open(os.path.join(results_path, 'best models.txt'), 'r') as f:
        best_model_paths = [ln.strip() for ln in f.readlines() if ln.strip()]

    state_dicts = [
        torch.load(os.path.join(results_path, rel), map_location='cpu')
        for rel in best_model_paths
    ]
    avg_state = {
        key: torch.stack([sd[key].float() for sd in state_dicts]).mean(0)
        for key in state_dicts[0]
    }
    print(f'  Averaged {len(state_dicts)} LOSO models as starting point.')
    log_write.write(f'Averaged {len(state_dicts)} LOSO models.\n\n')

    # ------------------------------------------------------------------ #
    # 4. Load golden data for catastrophic-forgetting check
    # ------------------------------------------------------------------ #
    X_golden, y_golden = load_golden_for_forgetting(
        dataset_conf['data_path'], window_size=window_size
    )

    # ------------------------------------------------------------------ #
    # 5. Baseline evaluation (no fine-tuning)
    # ------------------------------------------------------------------ #
    base_model = getModel(dataset_conf).to(device)
    base_model.load_state_dict(avg_state)

    baseline_child  = _evaluate_childclasses(
        base_model, X_eval, y_eval, pairs_eval, device)
    baseline_golden_preds = predict_batched(base_model, X_golden, device)
    baseline_golden_acc   = float(accuracy_score(y_golden, baseline_golden_preds))
    del base_model;  gc.collect()

    bl_info = (f'Baseline child-class acc (no fine-tuning): '
               f'{baseline_child["overall_acc"]*100:.2f}%  '
               f'F1: {baseline_child["overall_f1"]*100:.2f}%\n'
               f'Baseline golden acc                       : '
               f'{baseline_golden_acc*100:.2f}%\n')
    print(bl_info);  log_write.write(bl_info + '\n')
    print('Baseline per-pair:')
    for pair, d in baseline_child['per_pair'].items():
        print(f'  {pair}: {d["accuracy"]*100:.1f}%  ({d["n_windows"]} windows)')

    # ------------------------------------------------------------------ #
    # 6. Progressive fine-tuning loop
    # ------------------------------------------------------------------ #
    np.random.seed(random_seed)
    ft_curve      = []
    best_ft_acc   = -1.0
    best_ft_state = None
    best_ft_n_ids = None

    for n_ids_per_pair in range(1, max_ids_per_pair + 1):
        print(f"\n{'='*60}")
        print(f"  Fine-tuning: {n_ids_per_pair} ID(s) per pair  "
              f"(lr={finetune_lr:.1e}, epochs={finetune_epochs})")
        print(f"{'='*60}")
        log_write.write(f"\n{'='*50}\nn_ids_per_pair = {n_ids_per_pair}\n")

        # Stratified ID sampling — each pair independently
        sampled_ids = []
        for pair in sorted(pair_id_counts.keys()):
            p, c   = pair
            pmask  = (pairs_ft[:, 0] == p) & (pairs_ft[:, 1] == c)
            avail  = np.unique(ids_ft[pmask])
            n_take = min(n_ids_per_pair, len(avail))
            chosen = np.random.choice(avail, n_take, replace=False)
            sampled_ids.extend(chosen.tolist())

        pct_actual  = len(sampled_ids) / total_ft_ids * 100
        sample_mask = np.isin(ids_ft, sampled_ids)
        X_sub = X_ft[sample_mask]
        y_sub = y_ft[sample_mask]

        s_info = (f'  Sampled {len(sampled_ids)} IDs ({pct_actual:.1f}% of FT set) '
                  f'→ {len(X_sub)} windows  '
                  f'(class 0: {(y_sub==0).sum()}, class 1: {(y_sub==1).sum()})')
        print(s_info);  log_write.write(s_info + '\n')

        # Reload clean averaged weights (independent from all other steps)
        model     = getModel(dataset_conf).to(device)
        model.load_state_dict(avg_state)

        # Fine-tune ALL parameters with a low learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
        loss_fn   = nn.CrossEntropyLoss()
        train_dl  = DataLoader(
            TensorDataset(torch.tensor(X_sub),
                          torch.tensor(y_sub.astype(np.int64))),
            batch_size=min(batch_size, len(X_sub)),
            shuffle=True, num_workers=0
        )

        model.train()
        for epoch in range(finetune_epochs):
            ep_loss, n_b = 0.0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item();  n_b += 1
            if (epoch + 1) % 5 == 0:
                print(f'    Epoch {epoch+1:3d}  loss={ep_loss/n_b:.4f}')

        # Evaluate on held-out eval participant
        child_eval   = _evaluate_childclasses(model, X_eval, y_eval,
                                              pairs_eval, device)
        # Evaluate on golden data (catastrophic-forgetting check)
        golden_preds = predict_batched(model, X_golden, device)
        golden_acc   = float(accuracy_score(y_golden, golden_preds))
        golden_f1    = float(f1_score(y_golden, golden_preds,
                                      average='micro', zero_division=0))

        r_info = (f'\n  Child-class eval: acc={child_eval["overall_acc"]*100:.2f}%'
                  f'  F1={child_eval["overall_f1"]*100:.2f}%\n'
                  f'  Golden (core):    acc={golden_acc*100:.2f}%'
                  f'  F1={golden_f1*100:.2f}%  ← forgetting monitor\n'
                  f'  Per-pair accuracy:')
        print(r_info);  log_write.write(r_info + '\n')
        for pair, d in child_eval['per_pair'].items():
            line = f'    {pair}: {d["accuracy"]*100:.1f}%  ({d["n_windows"]} windows)'
            print(line);  log_write.write(line + '\n')

        ft_curve.append({
            'n_ids_per_pair': n_ids_per_pair,
            'n_ids_total':    len(sampled_ids),
            'pct':            pct_actual,
            'n_windows':      len(X_sub),
            'childclass_acc': child_eval['overall_acc'],
            'childclass_f1':  child_eval['overall_f1'],
            'golden_acc':     golden_acc,
            'golden_f1':      golden_f1,
            'per_pair':       child_eval['per_pair'],
        })

        if child_eval['overall_acc'] > best_ft_acc:
            best_ft_acc   = child_eval['overall_acc']
            best_ft_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ft_n_ids = n_ids_per_pair
            print('  ★ New best child-class accuracy — state saved')

        del model;  gc.collect()

    # ------------------------------------------------------------------ #
    # 7. Save best fine-tuned model
    # ------------------------------------------------------------------ #
    best_path = os.path.join(results_ft_path, 'best_finetuned_model.pt')
    torch.save(best_ft_state, best_path)
    summary = (f'\nBest fine-tuned model:\n'
               f'  IDs per pair    : {best_ft_n_ids}\n'
               f'  Child-class acc : {best_ft_acc*100:.2f}%\n'
               f'  Saved to        : {best_path}\n')
    print(summary);  log_write.write(summary)

    # ------------------------------------------------------------------ #
    # 8. Results table
    # ------------------------------------------------------------------ #
    hdr = (f"\n{'='*72}\n"
           f"{'IDs/pair':>9} {'%FT':>6} {'Windows':>8} "
           f"{'ChildAcc':>9} {'ChildF1':>8} {'GoldAcc':>8} {'GoldF1':>8}\n"
           f"{'='*72}")
    print(hdr);  log_write.write(hdr + '\n')
    for r in ft_curve:
        row = (f"{r['n_ids_per_pair']:>9}  {r['pct']:>5.1f}%  {r['n_windows']:>8}"
               f"  {r['childclass_acc']*100:>7.2f}%  {r['childclass_f1']*100:>6.2f}%"
               f"  {r['golden_acc']*100:>6.2f}%  {r['golden_f1']*100:>6.2f}%")
        print(row);  log_write.write(row + '\n')

    # ------------------------------------------------------------------ #
    # 9. Plots
    # ------------------------------------------------------------------ #
    plot_finetuning_curve(ft_curve, baseline_child['overall_acc'],
                          baseline_golden_acc, results_ft_path)
    plot_perpair_heatmap(ft_curve, baseline_child['per_pair'],
                         child_pairs, results_ft_path)

    log_write.close()
    print(f'\n  Fine-tuning results saved to: {results_ft_path}')


# ======================================================================
# Within-participant training / testing
# ======================================================================

def _draw_train_curve(train_losses, train_accs, results_path, participant):
    """Plot training loss and accuracy for a single within-participant run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_accs,   color='steelblue', label='Train')
    axes[0].set_title(f'Participant {participant} – Accuracy')
    axes[0].set_ylabel('Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].set_ylim(0, 1); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_losses, color='darkorange', label='Train')
    axes[1].set_title(f'Participant {participant} – Loss')
    axes[1].set_ylabel('Loss'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(results_path,
                             f'learning_curve_participant_{participant}.png'))
    plt.close(fig)


def train_test_within_participant(dataset_conf, test_data_path, results_wip_path,
                                  n_train_ac=7, n_train_bp=10,
                                  n_test_ac=1,  n_test_bp=2,
                                  epochs=50, batch_size=32, lr=1e-4,
                                  random_seed=42):
    """
    Train and evaluate a separate model for every participant using only that
    participant's own data.  No cross-subject information is used.

    For each participant the IDs are split by class:
      - Arm curl    : n_train_ac IDs for training, n_test_ac for testing
      - Bench press : n_train_bp IDs for training, n_test_bp for testing

    If a participant has fewer IDs than required (e.g. participant 17 has only
    6 arm curl IDs), the split is adjusted: n_test_ac / n_test_bp IDs are
    always reserved for testing and the remainder go to training.

    A fresh model is trained from scratch for each participant.  Results are
    logged per-participant and as an aggregate summary.

    Outputs (inside results_wip_path)
    ----------------------------------
    log_within_participant.txt
    learning_curve_participant_<pid>.png   – training loss / accuracy
    cm_subject_wip_<pid>.png              – confusion matrix
    test/Y_pred_Participant_<pid>.npy
    test/Y_true_Participant_<pid>.npy
    model_participant_<pid>.pt
    bar_Within-Participant Accuracy_per_subject.png
    bar_Within-Participant F1_per_subject.png
    """
    os.makedirs(results_wip_path, exist_ok=True)
    os.makedirs(os.path.join(results_wip_path, 'test'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log    = open(os.path.join(results_wip_path, 'log_within_participant.txt'), 'w')

    dataset        = dataset_conf.get('name')
    window_size    = dataset_conf.get('in_samples')
    classes_labels = dataset_conf.get('cl_labels')

    # ------------------------------------------------------------------ #
    # Load all OurData (raw, un-normalised)
    # ------------------------------------------------------------------ #
    X_all, y_all, _y_child, ids_all, parts_all, _pairs_arr, all_pairs = load_ourdata_all(
        test_data_path, dataset, window_size=window_size
    )

    hdr = (f"Within-participant training/testing\n"
           f"  Train split : {n_train_ac} arm curl IDs + {n_train_bp} bench press IDs\n"
           f"  Test split  : {n_test_ac}  arm curl ID  + {n_test_bp}  bench press IDs\n"
           f"  Epochs      : {epochs}  |  LR : {lr}  |  Batch : {batch_size}\n"
           f"  All pairs   : {all_pairs}\n")
    print(hdr); log.write(hdr + '\n')

    np.random.seed(random_seed)

    unique_participants = sorted(np.unique(parts_all).tolist())
    results = []

    for pid in unique_participants:
        print(f"\n{'='*60}\n  Participant {pid}\n{'='*60}")
        log.write(f"\n{'='*50}\nParticipant {pid}\n")

        pmask  = parts_all == pid
        X_pid  = X_all[pmask]
        y_pid  = y_all[pmask]
        ids_pid = ids_all[pmask]

        # IDs available per class for this participant
        ac_ids = np.unique(ids_pid[y_pid == 0])
        bp_ids = np.unique(ids_pid[y_pid == 1])

        id_info = (f"  Arm curl IDs    : {len(ac_ids)}\n"
                   f"  Bench press IDs : {len(bp_ids)}")
        print(id_info); log.write(id_info + '\n')

        # Require at least 1 train + 1 test per class
        if len(ac_ids) < n_test_ac + 1 or len(bp_ids) < n_test_bp + 1:
            msg = (f"  WARNING: not enough IDs for participant {pid} "
                   f"(AC={len(ac_ids)}, BP={len(bp_ids)}) — skipping.")
            print(msg); log.write(msg + '\n')
            continue

        # Adjust train count if fewer IDs than requested
        actual_train_ac = min(n_train_ac, len(ac_ids) - n_test_ac)
        actual_train_bp = min(n_train_bp, len(bp_ids) - n_test_bp)

        if actual_train_ac < n_train_ac:
            print(f"  Note: only {actual_train_ac} AC train IDs available "
                  f"(requested {n_train_ac})")
            log.write(f"  Adjusted AC train IDs: {actual_train_ac}\n")
        if actual_train_bp < n_train_bp:
            print(f"  Note: only {actual_train_bp} BP train IDs available "
                  f"(requested {n_train_bp})")
            log.write(f"  Adjusted BP train IDs: {actual_train_bp}\n")

        # Random shuffle then split
        ac_ids_shuf = ac_ids.copy(); np.random.shuffle(ac_ids_shuf)
        bp_ids_shuf = bp_ids.copy(); np.random.shuffle(bp_ids_shuf)

        train_ids = np.concatenate([ac_ids_shuf[:actual_train_ac],
                                    bp_ids_shuf[:actual_train_bp]])
        test_ids  = np.concatenate([ac_ids_shuf[actual_train_ac:actual_train_ac + n_test_ac],
                                    bp_ids_shuf[actual_train_bp:actual_train_bp + n_test_bp]])

        X_train = X_pid[np.isin(ids_pid, train_ids)]
        y_train = y_pid[np.isin(ids_pid, train_ids)]
        X_test  = X_pid[np.isin(ids_pid, test_ids)]
        y_test  = y_pid[np.isin(ids_pid, test_ids)]

        split_info = (f"  Train : {len(X_train)} windows "
                      f"(AC={( y_train==0).sum()}, BP={(y_train==1).sum()})\n"
                      f"  Test  : {len(X_test)} windows  "
                      f"(AC={(y_test==0).sum()},  BP={(y_test==1).sum()})")
        print(split_info); log.write(split_info + '\n')

        # Normalise: fit on training windows, apply to test
        N_tr, T, C = X_train.shape
        # scaler  = StandardScaler()
        # X_train = scaler.fit_transform(X_train.reshape(-1, C)).reshape(N_tr, T, C)
        N_te    = len(X_test)
        # X_test  = scaler.transform(X_test.reshape(-1, C)).reshape(N_te, T, C)

        X_train = X_train.reshape(N_tr, 1, T, C).astype(np.float32)
        X_test  = X_test.reshape(N_te,  1, T, C).astype(np.float32)

        # ---- Train model from scratch --------------------------------- #
        model = getModel(dataset_conf).to(device)

        cw_arr = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train)
        cw_t   = torch.tensor(cw_arr.astype(np.float32)).to(device)

        criterion = nn.CrossEntropyLoss(weight=cw_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_dl = DataLoader(
            TensorDataset(torch.tensor(X_train),
                          torch.tensor(y_train.astype(np.int64))),
            batch_size=min(batch_size, len(X_train)),
            shuffle=True, num_workers=0
        )

        train_losses, train_accs = [], []
        for epoch in range(epochs):
            model.train()
            ep_loss, correct, total = 0.0, 0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                correct += (logits.argmax(-1) == yb).sum().item()
                total   += len(yb)

            ep_acc  = correct / total
            ep_loss /= len(train_dl)
            train_losses.append(ep_loss)
            train_accs.append(ep_acc)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}  "
                      f"loss={ep_loss:.4f}  acc={ep_acc:.4f}")

        # ---- Evaluate ------------------------------------------------- #
        y_pred     = predict_batched(model, X_test, device)
        test_acc   = float(accuracy_score(y_test, y_pred))
        test_f1    = float(f1_score(y_test, y_pred,
                                    average='macro', zero_division=0))
        test_kappa = float(cohen_kappa_score(y_test, y_pred)
                           if len(np.unique(y_test)) > 1 else 0.0)

        # Per-class accuracy and F1
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        ac_mask  = y_test == 0
        bp_mask  = y_test == 1
        ac_acc   = float(accuracy_score(y_test[ac_mask], y_pred[ac_mask])) if ac_mask.any() else 0.0
        bp_acc   = float(accuracy_score(y_test[bp_mask], y_pred[bp_mask])) if bp_mask.any() else 0.0
        ac_f1    = float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0
        bp_f1    = float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0

        res_info = (f"\n  Arm Curl    — acc: {ac_acc:.4f}  f1: {ac_f1:.4f}\n"
                    f"  Bench Press — acc: {bp_acc:.4f}  f1: {bp_f1:.4f}\n"
                    f"  Overall     — acc: {test_acc:.4f}  f1: {test_f1:.4f}  kappa: {test_kappa:.4f}")
        print(res_info); log.write(res_info + '\n')

        report = classification_report(y_test, y_pred,
                                       target_names=classes_labels,
                                       zero_division=0)
        print(report); log.write(report + '\n')

        # ---- Save artefacts ------------------------------------------- #
        plot_confusion_matrix(y_test, y_pred, f'wip_{pid}',
                              results_wip_path, classes_labels)
        _draw_train_curve(train_losses, train_accs, results_wip_path, pid)
        np.save(os.path.join(results_wip_path, 'test',
                             f'Y_pred_Participant_{pid}.npy'), y_pred)
        np.save(os.path.join(results_wip_path, 'test',
                             f'Y_true_Participant_{pid}.npy'), y_test)
        torch.save(model.state_dict(),
                   os.path.join(results_wip_path, f'model_participant_{pid}.pt'))

        results.append({'participant': pid, 'n_train': N_tr, 'n_test': N_te,
                        'acc': test_acc, 'f1': test_f1, 'kappa': test_kappa,
                        'ac_acc': ac_acc, 'ac_f1': ac_f1,
                        'bp_acc': bp_acc, 'bp_f1': bp_f1,
                        'train_ac_ids': actual_train_ac,
                        'train_bp_ids': actual_train_bp})
        del model; gc.collect()

    if not results:
        print("No participants processed.");  log.close();  return

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    pids    = [r['participant'] for r in results]
    accs    = [r['acc']         for r in results]
    f1s     = [r['f1']          for r in results]
    kappas  = [r['kappa']       for r in results]
    ac_accs = [r['ac_acc']      for r in results]
    ac_f1s  = [r['ac_f1']       for r in results]
    bp_accs = [r['bp_acc']      for r in results]
    bp_f1s  = [r['bp_f1']       for r in results]

    summary = (f"\n{'='*60}\n"
               f"Within-Participant Results Summary ({len(results)} participants)\n"
               f"{'='*60}\n"
               f"  Arm Curl    — acc: {np.mean(ac_accs):.4f} ± {np.std(ac_accs):.4f}  "
               f"f1: {np.mean(ac_f1s):.4f} ± {np.std(ac_f1s):.4f}\n"
               f"  Bench Press — acc: {np.mean(bp_accs):.4f} ± {np.std(bp_accs):.4f}  "
               f"f1: {np.mean(bp_f1s):.4f} ± {np.std(bp_f1s):.4f}\n"
               f"  Overall     — acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}  "
               f"f1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  "
               f"kappa: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}\n\n"
               f"  Per-participant:\n")
    for r in results:
        summary += (f"    Participant {r['participant']:>3d}  "
                    f"AC(acc={r['ac_acc']:.4f} f1={r['ac_f1']:.4f})  "
                    f"BP(acc={r['bp_acc']:.4f} f1={r['bp_f1']:.4f})  "
                    f"Overall(acc={r['acc']:.4f} f1={r['f1']:.4f} kappa={r['kappa']:.4f})  "
                    f"[train AC={r['train_ac_ids']} BP={r['train_bp_ids']}]\n")
    print(summary); log.write(summary)

    draw_performance_barChart(pids, accs,   'Within-Participant Accuracy', results_wip_path)
    draw_performance_barChart(pids, f1s,    'Within-Participant F1',       results_wip_path)
    draw_performance_barChart(pids, kappas, 'Within-Participant Kappa',    results_wip_path)

    log.close()
    print(f"\n  Within-participant results saved to: {results_wip_path}")


# ======================================================================
# Within-participant 10-class (child-class) training / testing
# ======================================================================

def train_test_within_participant_childclass(dataset_conf, test_data_path,
                                             results_wip10_path,
                                             n_test_ids_per_pair=1,
                                             epochs=50, batch_size=32, lr=1e-4,
                                             random_seed=42):
    """
    Train and evaluate a 10-class model per participant using only that
    participant's own data.

    All 10 (parent, child) exercise pairs are treated as separate classes:
        (1,1) → 0   (1,2) → 1   (1,3) → 2   (1,4) → 3
        (5,5) → 4   (5,6) → 5   (5,7) → 6   (5,8) → 7   (5,9) → 8   (5,10) → 9

    ID split — for each (parent,child) pair independently:
        - Reserve n_test_ids_per_pair IDs for testing (default 1).
        - All remaining IDs go to training.
    Since each pair has only 2 IDs per participant this means 1 train + 1 test
    per pair (by default).  A participant is skipped if any pair has fewer
    than n_test_ids_per_pair + 1 IDs.

    A fresh 10-class PostFusion model is trained from scratch per participant.

    Outputs (inside results_wip10_path)
    ------------------------------------
    log_within_participant_10class.txt
    learning_curve_10class_participant_<pid>.png
    cm_10class_subject_<pid>.png
    test/Y_pred_10class_Participant_<pid>.npy
    test/Y_true_10class_Participant_<pid>.npy
    model_10class_participant_<pid>.pt
    bar_10-Class Accuracy_per_subject.png
    bar_10-Class F1_per_subject.png
    bar_10-Class Kappa_per_subject.png
    """
    os.makedirs(results_wip10_path, exist_ok=True)
    os.makedirs(os.path.join(results_wip10_path, 'test'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log    = open(os.path.join(results_wip10_path,
                               'log_within_participant_10class.txt'), 'w')

    dataset     = dataset_conf.get('name')
    window_size = dataset_conf.get('in_samples')

    # Build a 10-class dataset_conf for the model factory
    dataset_conf_10 = {**dataset_conf, 'n_classes': 10,
                       'cl_labels': OURDATA_10CLASS_LABELS}

    # ------------------------------------------------------------------ #
    # Load all OurData (raw, un-normalised)
    # ------------------------------------------------------------------ #
    X_all, _y_parent, y_child_all, ids_all, parts_all, pairs_arr_all, all_pairs = \
        load_ourdata_all(test_data_path, dataset, window_size=window_size)

    hdr = (f"Within-participant 10-class training/testing\n"
           f"  Classes     : {OURDATA_10CLASS_LABELS}\n"
           f"  Test IDs/pair : {n_test_ids_per_pair}\n"
           f"  All pairs   : {all_pairs}\n"
           f"  Epochs      : {epochs}  |  LR : {lr}  |  Batch : {batch_size}\n")
    print(hdr); log.write(hdr + '\n')

    np.random.seed(random_seed)

    unique_participants = sorted(np.unique(parts_all).tolist())
    results = []

    for pid in unique_participants:
        print(f"\n{'='*60}\n  Participant {pid}  [10-class]\n{'='*60}")
        log.write(f"\n{'='*50}\nParticipant {pid}  [10-class]\n")

        pmask     = parts_all == pid
        X_pid     = X_all[pmask]
        yc_pid    = y_child_all[pmask]
        ids_pid   = ids_all[pmask]
        pairs_pid = pairs_arr_all[pmask]   # (N, 2)

        # Collect unique (parent,child) pairs for this participant
        unique_pairs_pid = sorted(
            set(map(tuple, pairs_pid.tolist()))
        )

        # Verify all 10 expected pairs are present; warn if any missing
        expected = sorted(OURDATA_PAIR_TO_CLASS.keys())
        missing  = [p for p in expected if p not in unique_pairs_pid]
        if missing:
            msg = f"  Note: pairs {missing} absent for participant {pid}"
            print(msg); log.write(msg + '\n')

        # For each pair, check we have enough IDs to split
        train_ids_all, test_ids_all = [], []
        skip = False
        pair_id_info = []
        for pair in unique_pairs_pid:
            pair_mask  = (pairs_pid[:, 0] == pair[0]) & (pairs_pid[:, 1] == pair[1])
            pair_ids   = np.unique(ids_pid[pair_mask])
            n_ids      = len(pair_ids)
            if n_ids < n_test_ids_per_pair + 1:
                msg = (f"  WARNING: pair {pair} has only {n_ids} ID(s) "
                       f"(need ≥ {n_test_ids_per_pair + 1}) — skipping participant.")
                print(msg); log.write(msg + '\n')
                skip = True
                break
            shuffled = pair_ids.copy()
            np.random.shuffle(shuffled)
            test_ids_all.extend(shuffled[:n_test_ids_per_pair].tolist())
            train_ids_all.extend(shuffled[n_test_ids_per_pair:].tolist())
            pair_id_info.append(
                f"    {pair}: {n_ids} IDs → "
                f"{n_ids - n_test_ids_per_pair} train, {n_test_ids_per_pair} test"
            )

        if skip:
            continue

        id_info = '\n'.join(pair_id_info)
        print(id_info); log.write(id_info + '\n')

        train_mask = np.isin(ids_pid, train_ids_all)
        test_mask  = np.isin(ids_pid, test_ids_all)

        X_train = X_pid[train_mask]
        y_train = yc_pid[train_mask]
        X_test  = X_pid[test_mask]
        y_test  = yc_pid[test_mask]

        N_tr, T, C = X_train.shape
        N_te       = len(X_test)

        split_info = (f"  Train : {N_tr} windows across "
                      f"{len(np.unique(y_train))} child classes\n"
                      f"  Test  : {N_te} windows across "
                      f"{len(np.unique(y_test))} child classes")
        print(split_info); log.write(split_info + '\n')

        X_train = X_train.reshape(N_tr, 1, T, C).astype(np.float32)
        X_test  = X_test.reshape(N_te,  1, T, C).astype(np.float32)

        # ---- Train 10-class model from scratch ------------------------ #
        model = getModel(dataset_conf_10).to(device)

        cw_arr = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train)
        # Pad weight array to length 10 (some classes may be missing in tiny splits)
        full_cw = np.ones(10, dtype=np.float32)
        for cls, w in zip(np.unique(y_train), cw_arr):
            full_cw[cls] = w
        cw_t = torch.tensor(full_cw).to(device)

        criterion = nn.CrossEntropyLoss(weight=cw_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_dl = DataLoader(
            TensorDataset(torch.tensor(X_train),
                          torch.tensor(y_train.astype(np.int64))),
            batch_size=min(batch_size, len(X_train)),
            shuffle=True, num_workers=0,
        )

        train_losses, train_accs = [], []
        for epoch in range(epochs):
            model.train()
            ep_loss, correct, total = 0.0, 0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                correct += (logits.argmax(-1) == yb).sum().item()
                total   += len(yb)

            ep_acc  = correct / total
            ep_loss /= len(train_dl)
            train_losses.append(ep_loss)
            train_accs.append(ep_acc)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}  "
                      f"loss={ep_loss:.4f}  acc={ep_acc:.4f}")

        # ---- Evaluate (10-class + collapsed parent-class) ------------- #
        y_pred     = predict_batched(model, X_test, device)
        test_acc   = float(accuracy_score(y_test, y_pred))
        test_f1    = float(f1_score(y_test, y_pred,
                                    average='macro', zero_division=0))
        test_kappa = float(cohen_kappa_score(y_test, y_pred)
                           if len(np.unique(y_test)) > 1 else 0.0)

        # Collapse to parent classes for secondary metric
        y_test_par = np.array([OURDATA_10CLASS_TO_PARENT[c] for c in y_test])
        y_pred_par = np.array([OURDATA_10CLASS_TO_PARENT[c] for c in y_pred])
        par_acc    = float(accuracy_score(y_test_par, y_pred_par))
        par_f1     = float(f1_score(y_test_par, y_pred_par,
                                    average='macro', zero_division=0))

        res_info = (f"\n  10-class accuracy : {test_acc:.4f}\n"
                    f"  10-class macro F1 : {test_f1:.4f}\n"
                    f"  10-class kappa    : {test_kappa:.4f}\n"
                    f"  Parent accuracy   : {par_acc:.4f}\n"
                    f"  Parent macro F1   : {par_f1:.4f}")
        print(res_info); log.write(res_info + '\n')

        present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
        present_labels  = [OURDATA_10CLASS_LABELS[c] for c in present_classes]
        report = classification_report(y_test, y_pred,
                                       labels=present_classes,
                                       target_names=present_labels,
                                       zero_division=0)
        print(report); log.write(report + '\n')

        # ---- Save artefacts ------------------------------------------- #
        plot_confusion_matrix(y_test, y_pred, f'10class_{pid}',
                              results_wip10_path, OURDATA_10CLASS_LABELS)
        _draw_train_curve(train_losses, train_accs, results_wip10_path,
                          f'10class_{pid}')
        np.save(os.path.join(results_wip10_path, 'test',
                             f'Y_pred_10class_Participant_{pid}.npy'), y_pred)
        np.save(os.path.join(results_wip10_path, 'test',
                             f'Y_true_10class_Participant_{pid}.npy'), y_test)
        torch.save(model.state_dict(),
                   os.path.join(results_wip10_path,
                                f'model_10class_participant_{pid}.pt'))

        results.append({'participant': pid, 'n_train': N_tr, 'n_test': N_te,
                        'acc': test_acc, 'f1': test_f1, 'kappa': test_kappa,
                        'par_acc': par_acc, 'par_f1': par_f1})
        del model; gc.collect()

    if not results:
        print("No participants processed."); log.close(); return

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    pids   = [r['participant'] for r in results]
    accs   = [r['acc']         for r in results]
    f1s    = [r['f1']          for r in results]
    kappas = [r['kappa']       for r in results]
    par_accs = [r['par_acc']   for r in results]

    summary = (f"\n{'='*60}\n"
               f"Within-Participant 10-Class Results Summary ({len(results)} participants)\n"
               f"{'='*60}\n"
               f"  10-class Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}  "
               f"(range {min(accs):.4f}–{max(accs):.4f})\n"
               f"  10-class Macro F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n"
               f"  10-class Kappa    : {np.mean(kappas):.4f} ± {np.std(kappas):.4f}\n"
               f"  Parent Accuracy   : {np.mean(par_accs):.4f} ± {np.std(par_accs):.4f}\n\n"
               f"  Per-participant:\n")
    for r in results:
        summary += (f"    Participant {r['participant']:>3d}  "
                    f"10cls-acc={r['acc']:.4f}  f1={r['f1']:.4f}  "
                    f"kappa={r['kappa']:.4f}  par-acc={r['par_acc']:.4f}\n")
    print(summary); log.write(summary)

    draw_performance_barChart(pids, accs,     '10-Class Accuracy', results_wip10_path)
    draw_performance_barChart(pids, f1s,      '10-Class F1',       results_wip10_path)
    draw_performance_barChart(pids, kappas,   '10-Class Kappa',    results_wip10_path)
    draw_performance_barChart(pids, par_accs, 'Parent Accuracy (from 10-class)',
                              results_wip10_path)

    log.close()
    print(f"\n  10-class within-participant results saved to: {results_wip10_path}")


# ======================================================================
# LOOCV 10-class (leave-one-participant-out cross-validation)
# ======================================================================

def train_test_loso_10class(dataset_conf, data_path, results_path,
                             epochs=100, batch_size=32, lr=1e-4,
                             random_seed=42):
    """
    Train and evaluate a 10-class model using leave-one-participant-out
    cross-validation (LOOCV).  Each fold holds out one participant as the
    test set and trains on all remaining participants.

    Classes are the 10 (parent, child) exercise pairs:
        (1,1)→0  (1,2)→1  (1,3)→2  (1,4)→3
        (5,5)→4  (5,6)→5  (5,7)→6  (5,8)→7  (5,9)→8  (5,10)→9

    Normalisation: StandardScaler fit on training windows, applied to test.

    Outputs (inside results_path)
    --------------------------------
    log_loso_10class.txt
    learning_curves_loso10_subject_<pid>.png
    cm_loso10_subject_<pid>.png
    test/Y_pred_loso10_Sub_<pid>.npy
    test/Y_true_loso10_Sub_<pid>.npy
    bar_LOOCV 10-Class Accuracy_per_subject.png
    bar_LOOCV 10-Class F1_per_subject.png
    bar_LOOCV 10-Class Kappa_per_subject.png
    bar_LOOCV Parent Accuracy_per_subject.png
    """
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(os.path.join(results_path, 'test'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log    = open(os.path.join(results_path, 'log_loso_10class.txt'), 'w')

    dataset     = dataset_conf.get('name')
    window_size = dataset_conf.get('in_samples')

    dataset_conf_10 = {**dataset_conf, 'n_classes': 10,
                       'cl_labels': OURDATA_10CLASS_LABELS}

    # ------------------------------------------------------------------ #
    # Load all data once (raw, un-normalised)
    # ------------------------------------------------------------------ #
    X_all, _y_parent, y_child_all, _ids_all, parts_all, _pairs_arr, all_pairs = \
        load_ourdata_all(data_path, dataset, window_size=window_size)

    hdr = (f"LOOCV 10-class training/testing\n"
           f"  Classes   : {OURDATA_10CLASS_LABELS}\n"
           f"  All pairs : {all_pairs}\n"
           f"  Epochs    : {epochs}  |  LR : {lr}  |  Batch : {batch_size}\n")
    print(hdr); log.write(hdr + '\n')

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    unique_participants = sorted(np.unique(parts_all).tolist())
    n_sub   = len(unique_participants)
    results = []

    for fold_idx, test_pid in enumerate(unique_participants):
        print(f"\n{'='*60}")
        print(f"  LOOCV fold {fold_idx+1}/{n_sub}  —  Test participant: {test_pid}")
        print(f"{'='*60}")
        log.write(f"\n{'='*50}\nFold {fold_idx+1}/{n_sub}  —  Test participant {test_pid}\n")

        test_mask  = parts_all == test_pid
        train_mask = ~test_mask

        X_train_raw = X_all[train_mask]
        y_train     = y_child_all[train_mask].copy()
        X_test_raw  = X_all[test_mask]
        y_test      = y_child_all[test_mask].copy()

        N_tr, T, C = X_train_raw.shape
        N_te       = len(X_test_raw)

        split_info = (f"  Train: {N_tr} windows  ({len(unique_participants)-1} participants)\n"
                      f"  Test : {N_te} windows  (participant {test_pid})\n"
                      f"  Train classes present: {sorted(np.unique(y_train).tolist())}\n"
                      f"  Test  classes present: {sorted(np.unique(y_test).tolist())}")
        print(split_info); log.write(split_info + '\n')

        # Normalise: fit on train, apply to test
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw.reshape(-1, C)).reshape(N_tr, T, C)
        X_test  = scaler.transform(X_test_raw.reshape(-1, C)).reshape(N_te, T, C)

        X_train = X_train.reshape(N_tr, 1, T, C).astype(np.float32)
        X_test  = X_test.reshape(N_te,  1, T, C).astype(np.float32)

        # ---- Train ---------------------------------------------------- #
        model = getModel(dataset_conf_10).to(device)

        cw_arr   = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train)
        full_cw  = np.ones(10, dtype=np.float32)
        for cls, w in zip(np.unique(y_train), cw_arr):
            full_cw[cls] = w
        cw_t = torch.tensor(full_cw).to(device)

        criterion = nn.CrossEntropyLoss(weight=cw_t)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        train_dl = DataLoader(
            TensorDataset(torch.tensor(X_train),
                          torch.tensor(y_train.astype(np.int64))),
            batch_size=min(batch_size, N_tr),
            shuffle=True, num_workers=0,
        )

        train_losses, train_accs = [], []
        for epoch in range(epochs):
            model.train()
            ep_loss, correct, total = 0.0, 0, 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                correct += (logits.argmax(-1) == yb).sum().item()
                total   += len(yb)

            ep_acc  = correct / total
            ep_loss /= len(train_dl)
            train_losses.append(ep_loss)
            train_accs.append(ep_acc)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}  loss={ep_loss:.4f}  acc={ep_acc:.4f}")

        # ---- Evaluate ------------------------------------------------- #
        y_pred     = predict_batched(model, X_test, device)
        test_acc   = float(accuracy_score(y_test, y_pred))
        test_f1    = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        test_kappa = float(cohen_kappa_score(y_test, y_pred)
                           if len(np.unique(y_test)) > 1 else 0.0)

        # Collapse to parent classes
        y_test_par = np.array([OURDATA_10CLASS_TO_PARENT[c] for c in y_test])
        y_pred_par = np.array([OURDATA_10CLASS_TO_PARENT[c] for c in y_pred])
        par_acc    = float(accuracy_score(y_test_par, y_pred_par))
        par_f1     = float(f1_score(y_test_par, y_pred_par,
                                    average='macro', zero_division=0))

        # Per-child-class accuracy and F1
        per_cls_f1  = f1_score(y_test, y_pred, labels=list(range(10)),
                               average=None, zero_division=0)
        per_cls_acc = {}
        for c in range(10):
            mask_c = y_test == c
            if mask_c.any():
                per_cls_acc[c] = float((y_pred[mask_c] == c).mean())

        res_info = (f"\n  10-class acc  : {test_acc:.4f}  f1: {test_f1:.4f}"
                    f"  kappa: {test_kappa:.4f}\n"
                    f"  Parent   acc  : {par_acc:.4f}  f1: {par_f1:.4f}\n"
                    f"  Per child class:\n")
        for c in range(10):
            if c in per_cls_acc:
                res_info += (f"    {OURDATA_10CLASS_LABELS[c]:<8s}"
                             f"  acc={per_cls_acc[c]:.4f}"
                             f"  f1={per_cls_f1[c]:.4f}\n")
        print(res_info); log.write(res_info)

        present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
        present_labels  = [OURDATA_10CLASS_LABELS[c] for c in present_classes]
        report = classification_report(y_test, y_pred,
                                       labels=present_classes,
                                       target_names=present_labels,
                                       zero_division=0)
        print(report); log.write(report + '\n')

        # ---- Save artefacts ------------------------------------------- #
        plot_confusion_matrix(y_test, y_pred, f'loso10_{test_pid}',
                              results_path, OURDATA_10CLASS_LABELS)
        _draw_train_curve(train_losses, train_accs, results_path,
                          f'loso10_{test_pid}')
        np.save(os.path.join(results_path, 'test',
                             f'Y_pred_loso10_Sub_{test_pid}.npy'), y_pred)
        np.save(os.path.join(results_path, 'test',
                             f'Y_true_loso10_Sub_{test_pid}.npy'), y_test)

        results.append({
            'participant': test_pid,
            'n_train': N_tr, 'n_test': N_te,
            'acc': test_acc, 'f1': test_f1, 'kappa': test_kappa,
            'par_acc': par_acc, 'par_f1': par_f1,
            'per_cls_acc': per_cls_acc,
            'per_cls_f1': per_cls_f1,
        })
        del model; gc.collect()

    if not results:
        print("No folds processed."); log.close(); return

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    pids     = [r['participant'] for r in results]
    accs     = [r['acc']        for r in results]
    f1s      = [r['f1']         for r in results]
    kappas   = [r['kappa']      for r in results]
    par_accs = [r['par_acc']    for r in results]
    par_f1s  = [r['par_f1']     for r in results]

    summary = (f"\n{'='*60}\n"
               f"LOOCV 10-Class Results Summary ({len(results)} folds)\n"
               f"{'='*60}\n"
               f"  10-class acc  : {np.mean(accs):.4f} ± {np.std(accs):.4f}"
               f"  (range {min(accs):.4f}–{max(accs):.4f})\n"
               f"  10-class f1   : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n"
               f"  10-class kappa: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}\n"
               f"  Parent   acc  : {np.mean(par_accs):.4f} ± {np.std(par_accs):.4f}\n"
               f"  Parent   f1   : {np.mean(par_f1s):.4f} ± {np.std(par_f1s):.4f}\n\n"
               f"  Mean per-child-class acc / f1:\n")
    for c in range(10):
        fold_accs = [r['per_cls_acc'].get(c, float('nan')) for r in results]
        fold_f1s  = [r['per_cls_f1'][c] for r in results]
        valid_accs = [v for v in fold_accs if not np.isnan(v)]
        summary += (f"    {OURDATA_10CLASS_LABELS[c]:<8s}"
                    f"  acc={np.mean(valid_accs):.4f}"
                    f"  f1={np.mean(fold_f1s):.4f}\n")
    summary += "\n  Per-fold:\n"
    for r in results:
        summary += (f"    Sub {r['participant']:>3d}  "
                    f"10cls(acc={r['acc']:.4f} f1={r['f1']:.4f} κ={r['kappa']:.4f})  "
                    f"par(acc={r['par_acc']:.4f} f1={r['par_f1']:.4f})\n")
    print(summary); log.write(summary)

    draw_performance_barChart(pids, accs,     'LOOCV 10-Class Accuracy', results_path)
    draw_performance_barChart(pids, f1s,      'LOOCV 10-Class F1',       results_path)
    draw_performance_barChart(pids, kappas,   'LOOCV 10-Class Kappa',    results_path)
    draw_performance_barChart(pids, par_accs, 'LOOCV Parent Accuracy',   results_path)

    log.close()
    print(f"\n  LOOCV 10-class results saved to: {results_path}")


# ======================================================================
# Entry point
# ======================================================================

def run():
    dataset        = "Golden"
    in_samples     = 40
    n_channels     = 6
    n_classes      = 2
    classes_labels = ['arm curl', 'bench press'] #For 2 class parent classification
   # classes_labels = ['(1,1)', '(1,2)','(1,3)','(1,4)', '(5,5)', '(5,6)', '(5,7)', '(5,8)','(5,9)', '(5,10)']. # Uncomment for 10-class child classification 
    data_path      =  "/home/ec2-user/WHT_Team1A/OurData_v5.csv" #PATH to training data (golden dataset)
    test_data_path = "/home/ec2-user/WHT_Team1A/OurData_v5.csv" #PATH to test data (Our data, used for generalization test since we have no other dataset)
    print("DATA PATH: ", data_path)
    results_path = os.path.join(os.getcwd(), "results_golden_loso_acrossdataset2") #PATH to where results and checkpoints will be saved. Change "results_golden_loso_acrossdataset2" to your desired folder name. It will be created if it doesn't exist.
    print("RESULTS PATH: ", results_path)
    os.makedirs(results_path, exist_ok=True)
    print(f"Results will be saved to: {results_path}")
    
    results_gen_path = os.path.join(results_path, "generalization_test")
    os.makedirs(results_gen_path, exist_ok=True)
    print(f"Generalized results from our data will be saved to: {results_gen_path}")

    subjects = get_unique_subjects(data_path)
    print(f"Subjects found: {subjects}")

    dataset_conf = {
        'name':       dataset,
        'n_classes':  n_classes,
        'cl_labels':  classes_labels,
        'subjects':   subjects,
        'n_channels': n_channels,
        'in_samples': in_samples,
        'data_path':  data_path,
        'lr':         0.0001,   # used by fine_tune to set finetune_lr = lr / 10
    }

    train_conf = {
        'batch_size':  32,
        'epochs':      10,
        'lr':          0.0001,
        'LearnCurves': True,
        'n_train':     1,
    }

    train(dataset_conf, train_conf, results_path)

    model = getModel(dataset_conf)
    test(model, dataset_conf, results_path)
    test_general(dataset_conf, test_data_path, results_path, results_gen_path)
    


    ### The following sections each correspond to a different study as defined in the report. Uncomment the section(s) you wish to run.

    # ---- STUDY B: Progressive fine-tuning on child-class data -------------------- #
    # results_ft_path = os.path.join(results_path, "finetune_progressive")
    # print(f"\n{'='*60}")
    # print("Progressive fine-tuning (1 ID/pair → max IDs/pair)")
    # print(f"{'='*60}")
    # fine_tune(
    #     dataset_conf, test_data_path, results_path, results_ft_path,
    #     eval_participants=CHILD_EVAL_PARTICIPANTS,
    #     finetune_epochs=20,
    #     finetune_lr=1e-5,   # LOSO lr (1e-4) / 10
    #     batch_size=32,
    #     random_seed=42,
    # )

    #---- STUDY A: Within-participant training / testing (2-class parent) --------- #
    # results_wip_path = os.path.join(results_path, "within_participant2")
    # print(f"\n{'='*60}")
    # print("Within-participant training and testing (2-class parent)")
    # print(f"{'='*60}")
    # train_test_within_participant(
    #     dataset_conf, test_data_path, results_wip_path,
    #     n_train_ac=7, n_train_bp=10,
    #     n_test_ac=1,  n_test_bp=2,
    #     epochs=100, batch_size=32, lr=1e-4,
    #     random_seed=42,
    # )

    # ---- STUDY C:Within-participant training / testing (10-class child) ---------- #
    # results_wip10_path = os.path.join(results_path, "within_participant_childclass")
    # print(f"\n{'='*60}")
    # print("Within-participant training and testing (10-class child)")
    # print(f"{'='*60}")
    # train_test_within_participant_childclass(
    #     dataset_conf, test_data_path, results_wip10_path,
    #     n_test_ids_per_pair=1,
    #     epochs=100, batch_size=32, lr=1e-4,
    #     random_seed=42,
    # )

    # ---- LOOCV 10-class ------------------------------------------------- #
    # results_loocv10_path = os.path.join(results_path, "loocv_10class")
    # print(f"\n{'='*60}")
    # print("LOOCV 10-class training and testing")
    # print(f"{'='*60}")
    # train_test_loso_10class(
    #     dataset_conf, test_data_path, results_loocv10_path,
    #     epochs=100, batch_size=32, lr=1e-4,
    #     random_seed=42,
    # )


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPU(s) available: "
              f"{torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("No GPU found, running on CPU.")
    run()
