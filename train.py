import copy
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import config
from dataset import load_data, load_osats_data, get_loso_splits, get_louo_splits, LABEL_NAMES
from model import SurgicalFCN

REGRESSION_CHECKPOINT = os.path.join(config.MODEL_DIR, "best_model_regression.pth")
REGRESSION_EPOCHS = 1000  # per paper
REGRESSION_SEED = 42  # same as train_model


def _to_tensor(data):
    """Convert a (timesteps, 76) float32 array to a (1, 76, timesteps) tensor."""
    return torch.from_numpy(np.ascontiguousarray(data.T)).unsqueeze(0)


def train_model(train_data, num_classes=3, seed=42):
    """Train a SurgicalFCN on the given trials.

    Splits train_data 90/10 into train/val by shuffling with seed, trains for
    up to 1000 epochs with per-trial gradient updates, and returns the model
    checkpoint with the best validation loss.

    Args:
        train_data:  list of (data, label) tuples.
        num_classes: number of output classes.
        seed:        random seed for the train/val split and epoch shuffles.

    Returns:
        Trained SurgicalFCN loaded with the best-val-loss weights.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    indices = list(range(len(train_data)))
    random.shuffle(indices)
    n_val = max(1, int(0.1 * len(train_data)))
    val_split   = [train_data[i] for i in indices[:n_val]]
    train_split = [train_data[i] for i in indices[n_val:]]

    model = SurgicalFCN(num_classes=num_classes)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,            # paper §3.2
        betas=(0.9, 0.999),  # paper §3.2
        weight_decay=1e-5,   # paper §3.2
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, 1001):  # 1000 epochs: paper §3.2

        # --- train ---
        model.train()
        random.shuffle(train_split)
        train_loss = 0.0
        for data, label in train_split:
            x = _to_tensor(data)
            y = torch.tensor([label], dtype=torch.long)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_split)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, label in val_split:
                x = _to_tensor(data)
                y = torch.tensor([label], dtype=torch.long)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_split)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(best_state, os.path.join(config.MODEL_DIR, 'best_model.pth'))

        if epoch % 100 == 0:
            print(f'    Epoch {epoch:4d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

    model.load_state_dict(best_state)
    return model


def train_regression_model(train_data, num_outputs=6, seed=42,
                           checkpoint_path=REGRESSION_CHECKPOINT, verbose=True):
    torch.manual_seed(seed)
    random.seed(seed)

    indices = list(range(len(train_data)))
    random.shuffle(indices)
    n_val = max(1, int(0.1 * len(train_data)))
    val_split = [train_data[i] for i in indices[:n_val]]
    train_split = [train_data[i] for i in indices[n_val:]]

    model = SurgicalFCN(num_classes=num_outputs)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, REGRESSION_EPOCHS + 1):
        model.train()
        random.shuffle(train_split)
        train_loss = 0.0
        for data, osats in train_split:
            x = _to_tensor(data)
            y = torch.tensor(osats, dtype=torch.float32).unsqueeze(0)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_split)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, osats in val_split:
                x = _to_tensor(data)
                y = torch.tensor(osats, dtype=torch.float32).unsqueeze(0)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_split)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
            torch.save(best_state, checkpoint_path)

        if verbose and epoch % 100 == 0:
            print(f'    Epoch {epoch:4d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

    model.load_state_dict(best_state)
    return model



def predict(model, data):
    """Predict the skill class for a single trial.

    Args:
        model: trained SurgicalFCN.
        data:  numpy array of shape (timesteps, 76).

    Returns:
        Predicted class index as a Python int.
    """
    model.eval()
    with torch.no_grad():
        return model(_to_tensor(data)).argmax(dim=1).item()


def predict_regression(model, data):
    model.eval()
    with torch.no_grad():
        return model(_to_tensor(data)).squeeze(0).numpy()


def evaluate_accuracy(model, test_data):
    """Compute classification accuracy over a list of trials.

    Args:
        model:     trained SurgicalFCN.
        test_data: list of (data, label) tuples.

    Returns:
        Accuracy as a float in [0, 1].
    """
    correct = sum(predict(model, d) == l for d, l in test_data)
    return correct / len(test_data)


def evaluate_regression(model, test_data):
    all_abs = []
    all_sq = []
    for data, osats in test_data:
        pred = predict_regression(model, data)
        diff = pred - osats
        all_abs.append(np.abs(diff))
        all_sq.append(diff ** 2)
    all_abs = np.stack(all_abs)  # (n_trials, 6)
    all_sq = np.stack(all_sq)
    return {
        'mae': float(all_abs.mean()),
        'mse': float(all_sq.mean()),
        'mae_overall': float(all_abs[:, 4].mean()),
    }


def run_loso(dataset):
    """Run 5-fold Leave One Super-Trial Out cross-validation.

    Args:
        dataset: output of load_data.

    Returns:
        dict with 'mean_accuracy', 'fold_accuracies' (list) and 'fold_sizes'
        (list) — the shape consumed by evaluate.save_results.
    """
    splits = get_loso_splits(dataset)
    fold_accuracies, fold_sizes = [], []
    for i, (train, test) in enumerate(splits, 1):
        print(f'LOSO fold {i}/5 — training on {len(train)} trials...')
        model = train_model(train)
        acc = evaluate_accuracy(model, test)
        print(f'Fold {i}: {len(test)} trials, accuracy={acc:.1%}')
        fold_accuracies.append(acc)
        fold_sizes.append(len(test))
    mean_acc = sum(fold_accuracies) / len(fold_accuracies)
    print(f'LOSO mean accuracy: {mean_acc:.1%}')
    return {'mean_accuracy': mean_acc,
            'fold_accuracies': fold_accuracies,
            'fold_sizes': fold_sizes}


OSATS_NAMES = [
    "Respect for tissue", "Suture/needle handling", "Time and motion",
    "Flow of operation", "Overall performance", "Quality of final product",
]


def mean_spearman(gt, pred):
    """Mean Spearman's rho across the 6 OSATS targets, plus the per-target rhos.

    Mirrors the paper's regression metric: compute rho for each of the six
    predicted scores against ground truth, then average over the six.  A target
    whose predictions or ground truth are constant (rho undefined) is skipped in
    the mean and reported as nan.

    Args:
        gt:   (n_trials, 6) ground-truth OSATS scores.
        pred: (n_trials, 6) predicted OSATS scores.

    Returns:
        (mean_rho, per_target_rho) where per_target_rho is a length-6 list.
    """
    per_target = []
    for j in range(6):
        rho, _ = spearmanr(gt[:, j], pred[:, j])
        per_target.append(float(rho))
    valid = [r for r in per_target if not np.isnan(r)]
    mean_rho = float(np.mean(valid)) if valid else float('nan')
    return mean_rho, per_target


def run_loso_regression(dataset, num_outputs=6):
    """Run 5-fold Leave-One-Super-Trial-Out CV for the OSATS regression model.

    This is the regression analog of run_loso, following the paper, which uses a
    LOSO scheme for both the classification and regression tasks.  Each fold
    holds out all trials of one super-trial (trial number 1-5), trains a fresh
    SurgicalFCN on the rest, and predicts the held-out trials.  Held-out
    predictions are pooled across all five folds (each trial is predicted exactly
    once) and scored with the paper's metric: mean Spearman's rho over the six
    OSATS targets.  Per-fold checkpoints are saved as best_model_regression_fold{i}.pth
    so each can be exported and formally verified downstream.

    Args:
        dataset:     output of load_osats_data — (data, scores, subject, trial_num).
        num_outputs: number of OSATS targets (6).

    Returns:
        Dict with pooled 'mean_rho', 'per_target_rho', 'mae', and 'mae_overall'.
    """
    splits = get_loso_splits(dataset)  # yields (data, scores) pairs for regression
    all_gt, all_pred = [], []
    for i, (train, test) in enumerate(splits, 1):
        ckpt = os.path.join(config.MODEL_DIR, f'best_model_regression_fold{i}.pth')
        print(f'LOSO-reg fold {i}/5 — training on {len(train)} trials '
              f'(held-out super-trial {i}: {len(test)} trials)...')
        model = train_regression_model(train, num_outputs, checkpoint_path=ckpt, verbose=False)

        fold_gt = np.stack([o for _, o in test])
        fold_pred = np.stack([predict_regression(model, d) for d, _ in test])
        all_gt.append(fold_gt)
        all_pred.append(fold_pred)

        fold_mae = np.abs(fold_pred - fold_gt).mean()
        print(f'  fold {i}: MAE={fold_mae:.4f}  (checkpoint -> {ckpt})')

    gt = np.concatenate(all_gt)
    pred = np.concatenate(all_pred)
    mean_rho, per_target = mean_spearman(gt, pred)
    mae = float(np.abs(pred - gt).mean())
    mae_overall = float(np.abs(pred[:, 4] - gt[:, 4]).mean())

    print('\nPooled held-out results (paper metric = mean Spearman rho):')
    for name, rho in zip(OSATS_NAMES, per_target):
        print(f'  rho[{name:<24}] = {rho:.3f}')
    print(f'  mean rho (over 6 targets) = {mean_rho:.3f}   (paper Suturing FCN: 0.60)')
    print(f'  MAE (all dims)            = {mae:.4f}')
    print(f'  MAE (Overall Performance) = {mae_overall:.4f}')

    return {'mean_rho': mean_rho, 'per_target_rho': per_target,
            'mae': mae, 'mae_overall': mae_overall}


def run_louo(dataset):
    """Run 8-fold Leave One User Out cross-validation.

    Args:
        dataset: output of load_data.

    Returns:
        dict with 'mean_accuracy', 'subject_accuracies', 'subject_sizes' and
        'subject_labels' — the shape consumed by evaluate.save_results.
    """
    splits = get_louo_splits(dataset)
    subject_labels = {s: LABEL_NAMES[l] for _, l, s, _ in dataset}
    subject_accuracies, subject_sizes = {}, {}
    for subject, train, test in splits:
        print(f'LOUO subject {subject} — training on {len(train)} trials...')
        model = train_model(train)
        acc = evaluate_accuracy(model, test)
        print(f'Subject {subject}: {len(test)} trials, accuracy={acc:.1%}')
        subject_accuracies[subject] = acc
        subject_sizes[subject] = len(test)
    mean_acc = sum(subject_accuracies.values()) / len(subject_accuracies)
    print(f'LOUO mean accuracy: {mean_acc:.1%}')
    return {'mean_accuracy': mean_acc,
            'subject_accuracies': subject_accuracies,
            'subject_sizes': subject_sizes,
            'subject_labels': subject_labels}


if __name__ == '__main__':
    if '--regression' in sys.argv:
        dataset = load_osats_data()

        if '--single' in sys.argv:
            # Non-paper shortcut: train one model on ALL trials (in-sample MAE).
            # Kept only for quick experiments; does NOT follow the paper's LOSO.
            train_data = [(d, o) for d, o, s, t in dataset]
            print(f'[--single] Training one regression model on all {len(train_data)} trials...')
            model = train_regression_model(train_data)
            metrics = evaluate_regression(model, train_data)
            print(f'In-sample MAE (all dims): {metrics["mae"]:.4f}')
            print(f'In-sample MAE (Overall Performance): {metrics["mae_overall"]:.4f}')
            print(f'Weights saved to {REGRESSION_CHECKPOINT}')
            sys.exit(0)

        # Paper-faithful default: LOSO cross-validation for the regression task.
        print(f'Regression LOSO on {len(dataset)} Suturing trials '
              f'(leave-one-super-trial-out, per paper)...\n')
        run_loso_regression(dataset)
        sys.exit(0)

    dataset = load_data()

    labels = [l for _, l, _, _ in dataset]
    label_names = {0: 'Novice', 1: 'Intermediate', 2: 'Expert'}
    counts = Counter(labels)
    print(f'Total trials: {len(dataset)}')
    for cls in sorted(label_names):
        print(f'  {label_names[cls]}: {counts[cls]}')
    print()

    loso_results = run_loso(dataset)
    print()
    louo_results = run_louo(dataset)

    # Reuse the rich formatter so train.py and evaluate.py emit the identical
    # results file (no more competing simple/rich versions clobbering each other).
    # Imported here, not at module top, to avoid a train<->evaluate import cycle.
    from evaluate import save_results
    save_results(loso_results, louo_results)
    print(f'\nResults saved to {os.path.join(config.RESULTS_DIR, "results.txt")}')
