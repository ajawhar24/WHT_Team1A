import os
import gc
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.utils import class_weight, shuffle as sk_shuffle
from sklearn.metrics import classification_report

import models
from preprocess_golden_v3 import get_data, get_unique_subjects, load_new_dataset

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
    f1       = f1_score(y_true, y_pred, average='macro')
    title    = (f"Sub: {sub}  Macro F1: {round(f1*100, 2)}"
                f"  Accuracy: {round(accuracy*100, 2)}")

    cm = confusion_matrix(y_true, y_pred)
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
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects], rotation=45)
    ax.set_title(f'Model {label} per subject (LOSO)')
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(os.path.join(results_path, f"bar_{label.lower()}_per_subject.png"))
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
    plot_confusion_matrix(labels_all, predic_all, "All", results_path, classes_labels)

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
    X_all, y_all, subject_ids = load_new_dataset(
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
    f1_per_sub   = np.zeros(n_sub)
    kappa_per_sub = np.zeros(n_sub)
    labels_all    = np.array([])
    predic_all    = np.array([])

    for s_idx, sub in enumerate(subjects):
        mask   = subject_ids == sub
        X_sub  = X_all[mask]
        y_sub  = y_all[mask]
        y_pred = ensemble_predict(X_sub)

        f1_per_sub[s_idx]   = f1_score(y_sub, y_pred)
        kappa_per_sub[s_idx] = cohen_kappa_score(y_sub, y_pred)

        plot_confusion_matrix(y_sub, y_pred, sub, results_gen_path, classes_labels)
        np.save(os.path.join(results_gen_path, "test", f"Y_truth_Sub_{sub}.npy"), y_sub)
        np.save(os.path.join(results_gen_path, "test", f"Y_pred_Sub_{sub}.npy"),  y_pred)

        labels_all = np.concatenate((labels_all, y_sub))
        predic_all = np.concatenate((predic_all, y_pred))

        info = (f'Subject: {sub}'
                f'   acc: {f1_per_sub[s_idx]:.4f}'
                f'   kappa: {kappa_per_sub[s_idx]:.4f}')
        print(info)
        log_write.write(info + '\n')

    # ---- Overall summary ----------------------------------------------- #
    info = (f'\nEnsemble results on new dataset ({n_sub} subjects):\n'
            f'Accuracy = {np.mean(f1_per_sub):.4f} ± {f1_per_sub.std():.4f}\n'
            f'Kappa    = {np.mean(kappa_per_sub):.4f} ± {kappa_per_sub.std():.4f}\n')
    print(info)
    log_write.write(info)

    draw_performance_barChart(subjects, f1_per_sub,   'F1 Score', results_gen_path)
    draw_performance_barChart(subjects, kappa_per_sub, 'Kappa',    results_gen_path)
    plot_confusion_matrix(labels_all, predic_all, "All", results_gen_path, classes_labels)

    print(classification_report(labels_all, predic_all, target_names=classes_labels))
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
        n_windows     = 4,
        F1            = 32,
        D             = 4,
        kernelSize    = 20,
        dropout       = 0.1,
        di_kernelSize = 3,
        di_filters    = 32,
        di_dropout    = 0.1,
        di_activation = 'elu'
    )
    return model


# ======================================================================
# Entry point
# ======================================================================

def run():
    dataset        = "Golden"
    in_samples     = 40
    n_channels     = 6
    n_classes      = 2
    classes_labels = ['arm curl', 'bench press']
    data_path      = "/home/ec2-user/WHT_Team1A/golden_dataset_v4.csv"
    test_data_path = "/home/ec2-user/WHT_Team1A/OurData.csv"
    print("DATA PATH: ", data_path)
    results_path = os.path.join(os.getcwd(), "results_golden_loso")
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
    }

    train_conf = {
        'batch_size':  32,
        'epochs':      35,
        'lr':          0.0001,
        'LearnCurves': True,
        'n_train':     1,
    }

    # train(dataset_conf, train_conf, results_path)

    # model = getModel(dataset_conf)
    # test(model, dataset_conf, results_path)
    test_general(dataset_conf, test_data_path, results_path, results_gen_path)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPU(s) available: "
              f"{torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("No GPU found, running on CPU.")
    run()
