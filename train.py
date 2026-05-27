import copy
import datetime
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from dataset import load_data, get_loso_splits, get_louo_splits
from model import SurgicalFCN


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
            torch.save(best_state, 'best_model.pth')

        if epoch % 100 == 0:
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


def run_loso(dataset):
    """Run 5-fold Leave One Super-Trial Out cross-validation.

    Args:
        dataset: output of load_data.

    Returns:
        Mean accuracy across all 5 folds as a float in [0, 1].
    """
    splits = get_loso_splits(dataset)
    accuracies = []
    for i, (train, test) in enumerate(splits, 1):
        print(f'LOSO fold {i}/5 — training on {len(train)} trials...')
        model = train_model(train)
        acc = evaluate_accuracy(model, test)
        print(f'Fold {i}: {len(test)} trials, accuracy={acc:.1%}')
        accuracies.append(acc)
    mean_acc = sum(accuracies) / len(accuracies)
    print(f'LOSO mean accuracy: {mean_acc:.1%}')
    return mean_acc


def run_louo(dataset):
    """Run 8-fold Leave One User Out cross-validation.

    Args:
        dataset: output of load_data.

    Returns:
        Mean accuracy across all 8 folds as a float in [0, 1].
    """
    splits = get_louo_splits(dataset)
    accuracies = []
    for subject, train, test in splits:
        print(f'LOUO subject {subject} — training on {len(train)} trials...')
        model = train_model(train)
        acc = evaluate_accuracy(model, test)
        print(f'Subject {subject}: {len(test)} trials, accuracy={acc:.1%}')
        accuracies.append(acc)
    mean_acc = sum(accuracies) / len(accuracies)
    print(f'LOUO mean accuracy: {mean_acc:.1%}')
    return mean_acc


if __name__ == '__main__':
    dataset = load_data('data')

    labels = [l for _, l, _, _ in dataset]
    label_names = {0: 'Novice', 1: 'Intermediate', 2: 'Expert'}
    counts = Counter(labels)
    print(f'Total trials: {len(dataset)}')
    for cls in sorted(label_names):
        print(f'  {label_names[cls]}: {counts[cls]}')
    print()

    loso_acc = run_loso(dataset)
    print()
    louo_acc = run_louo(dataset)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('results.txt', 'w') as f:
        f.write(f'Results — {timestamp}\n')
        f.write(f'LOSO mean accuracy: {loso_acc:.1%}\n')
        f.write(f'LOUO mean accuracy: {louo_acc:.1%}\n')
    print(f'\nResults saved to results.txt')
