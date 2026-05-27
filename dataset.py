import os
import numpy as np

LABEL_MAP = {'N': 0, 'I': 1, 'E': 2}
LABEL_NAMES = {0: 'Novice', 1: 'Intermediate', 2: 'Expert'}

# Within each 19-column block, reorder from raw JIGSAWS layout
# [xyz(0-2), R(3-11), linvel(12-14), rotvel(15-17), gripper(18)]
# to paper layout [xyz, linvel, rotvel, R, gripper].
BLOCK_REORDER = [0, 1, 2, 12, 13, 14, 15, 16, 17, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18]
BLOCK_SIZE = 19
N_BLOCKS = 4


def reorder_columns(data):
    """Apply within-block column reordering to all 4 manipulator blocks.

    Converts raw JIGSAWS column order to the layout used by Ismail Fawaz et al.
    (arXiv:1908.07319): [xyz, linvel, rotvel, R, gripper] per block.

    Args:
        data: numpy array of shape (timesteps, 76).

    Returns:
        Reordered numpy array of the same shape (timesteps, 76).
    """
    cols = []
    for b in range(N_BLOCKS):
        offset = b * BLOCK_SIZE
        cols.extend(offset + idx for idx in BLOCK_REORDER)
    return data[:, cols]


def load_data(data_dir):
    """Load JIGSAWS Suturing kinematics and skill labels.

    Reads meta_file_Suturing.txt to determine which trials exist and their
    labels, then loads each corresponding kinematic .txt file from
    Suturing/kinematics/AllGestures/.  Column reordering is applied via
    reorder_columns before the array is stored.

    Args:
        data_dir: path to the dataset root, the directory that contains the
                  Suturing/ subdirectory (e.g. 'data').

    Returns:
        List of (data, label, subject, trial_num) tuples where:
          - data      is a float32 array of shape (timesteps, 76)
          - label     is an int  (0=Novice, 1=Intermediate, 2=Expert)
          - subject   is a single-character string (e.g. 'B')
          - trial_num is an int (1–5)
    """
    suturing_dir = os.path.join(data_dir, 'Suturing')
    meta_path = os.path.join(suturing_dir, 'meta_file_Suturing.txt')
    kin_dir = os.path.join(suturing_dir, 'kinematics', 'AllGestures')

    label_dict = {}
    with open(meta_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            name, label_str = parts[0], parts[1]
            label_dict[name] = LABEL_MAP[label_str]

    dataset = []
    for name, label in sorted(label_dict.items()):
        kin_path = os.path.join(kin_dir, name + '.txt')
        if not os.path.exists(kin_path):
            print(f'Warning: kinematic file not found, skipping: {kin_path}')
            continue
        data = np.loadtxt(kin_path, dtype=np.float32)
        data = reorder_columns(data)
        subject = name[9]           # 'Suturing_B001' -> 'B'
        trial_num = int(name[10:])  # 'Suturing_B001' -> 1
        dataset.append((data, label, subject, trial_num))

    return dataset


def get_loso_splits(dataset):
    """Build Leave One Super-Trial Out cross-validation splits.

    Holds out all trials with a given trial number (1 through 5) in turn,
    using the remaining trials for training.

    Args:
        dataset: list of (data, label, subject, trial_num) tuples as
                 returned by load_data.

    Returns:
        List of 5 (train, test) tuples, one per held-out trial number.
        Each of train and test is a list of (data, label) tuples.
    """
    splits = []
    for held_out in range(1, 6):
        train = [(d, l) for d, l, s, t in dataset if t != held_out]
        test  = [(d, l) for d, l, s, t in dataset if t == held_out]
        splits.append((train, test))
    return splits


def get_louo_splits(dataset):
    """Build Leave One User Out cross-validation splits.

    Holds out all trials from one subject at a time, training on the
    remaining subjects.

    Args:
        dataset: list of (data, label, subject, trial_num) tuples as
                 returned by load_data.

    Returns:
        List of 8 (subject, train, test) tuples, one per subject in sorted
        order.  train and test are lists of (data, label) tuples.
    """
    subjects = sorted({s for _, _, s, _ in dataset})
    splits = []
    for subject in subjects:
        train = [(d, l) for d, l, s, t in dataset if s != subject]
        test  = [(d, l) for d, l, s, t in dataset if s == subject]
        splits.append((subject, train, test))
    return splits


if __name__ == '__main__':
    dataset = load_data('data')

    print(f'Total trials loaded: {len(dataset)}')
    print()
    for data, label, subject, trial_num in dataset:
        print(f'  Subject={subject}  Trial={trial_num}  Shape={data.shape}  Label={LABEL_NAMES[label]}')
    print()

    loso = get_loso_splits(dataset)
    print(f'LOSO: {len(loso)} folds')
    for i, (train, test) in enumerate(loso, 1):
        print(f'  Fold {i} (trial {i} held out): train={len(train)}  test={len(test)}')
    print()

    louo = get_louo_splits(dataset)
    print(f'LOUO: {len(louo)} folds')
    for subject, train, test in louo:
        print(f'  Subject={subject}: train={len(train)}  test={len(test)}')
