import datetime
import os
from collections import Counter

import torch

import config
from dataset import load_data, get_loso_splits, get_louo_splits
from train import train_model, evaluate_accuracy

LABEL_NAMES = {0: 'Novice', 1: 'Intermediate', 2: 'Expert'}


def print_dataset_summary(dataset):
    """Print trial counts, class distribution, and per-subject breakdown.

    Args:
        dataset: list of (data, label, subject, trial_num) tuples from load_data.
    """
    label_counts = Counter(l for _, l, _, _ in dataset)

    subject_info = {}
    for _, l, s, _ in dataset:
        if s not in subject_info:
            subject_info[s] = {'label': LABEL_NAMES[l], 'count': 0}
        subject_info[s]['count'] += 1

    print('Dataset summary')
    print(f'  Total trials: {len(dataset)}')
    print('  Class distribution:')
    for cls in sorted(LABEL_NAMES):
        print(f'    {LABEL_NAMES[cls]:<14}: {label_counts[cls]} trials')
    print('  Subject breakdown:')
    for s in sorted(subject_info):
        info = subject_info[s]
        print(f'    Subject {s}: {info["label"]:<14}  {info["count"]} trials')
    n_novice       = sum(1 for v in subject_info.values() if v['label'] == 'Novice')
    n_intermediate = sum(1 for v in subject_info.values() if v['label'] == 'Intermediate')
    n_expert       = sum(1 for v in subject_info.values() if v['label'] == 'Expert')
    print(f'  Class imbalance: {n_novice} Novice subjects, '
          f'{n_intermediate} Intermediate, {n_expert} Expert')


def save_results(loso_results, louo_results, filepath=os.path.join(config.RESULTS_DIR, 'results.txt')):
    """Write a human-readable results file.

    Args:
        loso_results: dict with keys mean_accuracy (float), fold_accuracies
                      (list of 5 floats), fold_sizes (list of 5 ints).
        louo_results: dict with keys mean_accuracy (float), subject_accuracies
                      (dict subject→float), subject_sizes (dict subject→int),
                      subject_labels (dict subject→skill-level string).
        filepath:     output path (default 'results/results.txt').
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Reconstruct class-level trial counts from louo breakdown
    label_trial_counts = Counter()
    for s, label_name in louo_results['subject_labels'].items():
        label_trial_counts[label_name] += louo_results['subject_sizes'][s]
    total_trials = sum(louo_results['subject_sizes'].values())

    lines = [
        'Surgical Skill Assessment — FCN Results (Ismail Fawaz et al., arXiv:1908.07319)',
        f'Generated: {timestamp}',
        '',
        'Dataset',
        f'  Total trials: {total_trials}',
    ]
    for label_name in ('Novice', 'Intermediate', 'Expert'):
        lines.append(f'  {label_name:<14}: {label_trial_counts[label_name]} trials')

    lines += [
        '',
        'LOSO Cross-Validation (Leave One Super-Trial Out)',
    ]
    for i, (acc, n) in enumerate(
        zip(loso_results['fold_accuracies'], loso_results['fold_sizes']), 1
    ):
        lines.append(f'  Fold {i} ({n:2d} trials): {acc:.1%}')
    lines.append(f'  Mean:              {loso_results["mean_accuracy"]:.1%}')
    lines.append(f'  Target (paper):    100%')

    lines += [
        '',
        'LOUO Cross-Validation (Leave One User Out)',
    ]
    for s in sorted(louo_results['subject_accuracies']):
        acc   = louo_results['subject_accuracies'][s]
        n     = louo_results['subject_sizes'][s]
        label = louo_results['subject_labels'][s]
        lines.append(f'  Subject {s} - {label:<14} ({n} trials): {acc:.1%}')
    lines.append(f'  Mean: {louo_results["mean_accuracy"]:.1%}')

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    torch.manual_seed(42)

    dataset = load_data()
    print_dataset_summary(dataset)
    print()
    print('This will take a while — training 13 models (5 LOSO + 8 LOUO) each for up to 1000 epochs')
    print()

    # --- LOSO ---
    loso_splits = get_loso_splits(dataset)
    fold_accuracies = []
    fold_sizes = []
    for i, (train, test) in enumerate(loso_splits, 1):
        print(f'LOSO fold {i}/5 — training on {len(train)} trials...')
        model = train_model(train)
        acc = evaluate_accuracy(model, test)
        print(f'  Fold {i}: {len(test)} trials, accuracy={acc:.1%}')
        fold_accuracies.append(acc)
        fold_sizes.append(len(test))
    loso_mean = sum(fold_accuracies) / len(fold_accuracies)

    loso_results = {
        'mean_accuracy':   loso_mean,
        'fold_accuracies': fold_accuracies,
        'fold_sizes':      fold_sizes,
    }

    print()

    # --- LOUO ---
    louo_splits = get_louo_splits(dataset)
    subject_label_map = {s: LABEL_NAMES[l] for _, l, s, _ in dataset}
    subject_accuracies = {}
    subject_sizes = {}
    for subject, train, test in louo_splits:
        print(f'LOUO subject {subject} — training on {len(train)} trials...')
        model = train_model(train)
        acc = evaluate_accuracy(model, test)
        print(f'  Subject {subject}: {len(test)} trials, accuracy={acc:.1%}')
        subject_accuracies[subject] = acc
        subject_sizes[subject] = len(test)
    louo_mean = sum(subject_accuracies.values()) / len(subject_accuracies)

    louo_results = {
        'mean_accuracy':      louo_mean,
        'subject_accuracies': subject_accuracies,
        'subject_sizes':      subject_sizes,
        'subject_labels':     subject_label_map,
    }

    # --- Results table ---
    print()
    print('===== RESULTS =====')
    print()
    print('LOSO Cross-Validation:')
    for i, (acc, n) in enumerate(zip(fold_accuracies, fold_sizes), 1):
        print(f'  Fold {i} ({n} trials):{acc:>8.1%}')
    print(f'  Mean: {loso_mean:.1%}')
    print(f'  Target (paper): 100%')
    print()
    print('LOUO Cross-Validation:')
    for s in sorted(subject_accuracies):
        acc   = subject_accuracies[s]
        n     = subject_sizes[s]
        label = subject_label_map[s]
        print(f'  Subject {s} - {label:<14} ({n} trials): {acc:>6.1%}')
    print(f'  Mean: {louo_mean:.1%}')

    save_results(loso_results, louo_results)
    print()
    print('Results saved to results/results.txt')
