import os
import re
import numpy as np
import pandas as pd


# Within each block of 20 age-sorted subjects, these positions go to val/test.
# The rest (14/20 = 70%) go to train.
# Positions are spread across the block so each split covers the full age range.
#   val  at ~15%, 50%, 85% of each block  -> positions {3, 10, 17}
#   test at ~30%, 65%, 95% of each block  -> positions {6, 13, 19}
_CYCLE        = 20
_VAL_OFFSETS  = {3, 10, 17}   # 3/20 = 15%
_TEST_OFFSETS = {6, 13, 19}   # 3/20 = 15%


def data_split_ixi(nii_dir, xls_path):
    """
    Build train / validation / test splits for the IXI T1 dataset.

    Subjects are sorted by age, then assigned systematically so that each
    split contains subjects spanning the full age range (70 / 15 / 15%).

    Args:
        nii_dir  (str): Directory containing IXI T1 NIfTI files.
        xls_path (str): Path to IXI.xls demographics file.

    Returns:
        dict with keys:
            'train_files' / 'train_ages' / 'train_ids'
            'val_files'   / 'val_ages'   / 'val_ids'
            'test_files'  / 'test_ages'  / 'test_ids'
        Each *_files value is a list of full file paths.
        Each *_ages  value is a numpy array of floats.
        Each *_ids   value is a numpy array of ints (IXI_ID).
    """
    # --- load demographics, drop subjects without age ---
    demo = pd.read_excel(xls_path, usecols=['IXI_ID', 'AGE'])
    demo = demo.dropna(subset=['AGE']).copy()
    demo['IXI_ID'] = demo['IXI_ID'].astype(int)
    id_to_age = dict(zip(demo['IXI_ID'], demo['AGE']))

    # --- match NIfTI files to IXI IDs using regex on filename ---
    nii_files = sorted(
        f for f in os.listdir(nii_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    )

    matched = []
    for fname in nii_files:
        m = re.match(r'IXI(\d+)', fname)
        if m:
            ixi_id = int(m.group(1))
            if ixi_id in id_to_age:
                matched.append((ixi_id, id_to_age[ixi_id], os.path.join(nii_dir, fname)))

    if not matched:
        raise ValueError("No subjects matched between NIfTI files and demographics.")

    print(f"Matched subjects (file + valid age): {len(matched)}")

    # --- sort by age ---
    matched.sort(key=lambda x: x[1])

    # --- assign splits using the cyclic rule ---
    train_files, train_ages, train_ids = [], [], []
    val_files,   val_ages,   val_ids   = [], [], []
    test_files,  test_ages,  test_ids  = [], [], []

    for i, (ixi_id, age, fpath) in enumerate(matched):
        offset = i % _CYCLE
        if offset in _VAL_OFFSETS:
            val_files.append(fpath);   val_ages.append(age);   val_ids.append(ixi_id)
        elif offset in _TEST_OFFSETS:
            test_files.append(fpath);  test_ages.append(age);  test_ids.append(ixi_id)
        else:
            train_files.append(fpath); train_ages.append(age); train_ids.append(ixi_id)

    return {
        'train_files': train_files, 'train_ages': np.array(train_ages), 'train_ids': np.array(train_ids),
        'val_files':   val_files,   'val_ages':   np.array(val_ages),   'val_ids':   np.array(val_ids),
        'test_files':  test_files,  'test_ages':  np.array(test_ages),  'test_ids':  np.array(test_ids),
    }


if __name__ == '__main__':
    NII_DIR  = '/Users/tanchaud/Data_IXI/IXI-T1'
    XLS_PATH = '/Users/tanchaud/Data_IXI/IXI.xls'

    splits = data_split_ixi(NII_DIR, XLS_PATH)

    n_train = len(splits['train_files'])
    n_val   = len(splits['val_files'])
    n_test  = len(splits['test_files'])
    n_total = n_train + n_val + n_test

    print(f"\nSplit summary (total: {n_total})")
    print(f"  Train : {n_train:3d}  ({100*n_train/n_total:.1f}%)"
          f"  age {splits['train_ages'].min():.1f} - {splits['train_ages'].max():.1f}")
    print(f"  Val   : {n_val:3d}  ({100*n_val/n_total:.1f}%)"
          f"  age {splits['val_ages'].min():.1f} - {splits['val_ages'].max():.1f}")
    print(f"  Test  : {n_test:3d}  ({100*n_test/n_total:.1f}%)"
          f"  age {splits['test_ages'].min():.1f} - {splits['test_ages'].max():.1f}")
