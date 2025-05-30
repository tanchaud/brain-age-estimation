import numpy as np

def data_split(mri_data_list, ages, rids):
    """
    Sorts subjects by age and splits them into training and test sets (3:1 ratio).

    Args:
        mri_data_list (list): A list of MRI image data (e.g., NumPy arrays).
                              This list should correspond to the order of rids before sorting.
        ages (np.array or list): Array of ages for each subject.
        rids (np.array or list): Array of research IDs for each subject.

    Returns:
        dict: A dictionary containing training and testing sets for images,
              labels (ages), and RIDs.
              Keys: 'TS' (Training Set images), 'TS_labels' (Training Set ages),
                    'TS_Rid' (Training Set RIDs), 'VS' (Validation/Test Set images),
                    'VS_labels' (Validation/Test Set ages), 'VS_Rid' (Validation/Test Set RIDs).
    """
    ages = np.array(ages)
    rids = np.array(rids)

    # Sort by age
    idx_sorted = np.argsort(ages)
    sorted_ages = ages[idx_sorted]
    sorted_rids = rids[idx_sorted]

    # Pre-sort mri_data_list based on the original rids to match the sorted_rids
    # This requires a mapping from original rids to their mri_data
    # Assuming mri_data_list is initially in the same order as the input 'rids'
    # and 'ages' before sorting.
    # Create a dictionary to map original RIDs to their MRI data
    rid_to_mri_map = {rid: mri for rid, mri in zip(rids, mri_data_list)}

    train_rids = []
    train_labels = []
    test_rids = []
    test_labels = []

    for i in range(len(sorted_rids)):
        if (i + 1) % 4 == 0:  # Test set (every 4th subject)
            test_rids.append(sorted_rids[i])
            test_labels.append(sorted_ages[i])
        else:  # Training set
            train_rids.append(sorted_rids[i])
            train_labels.append(sorted_ages[i])

    # Retrieve MRI data for training and test sets
    train_images = []
    for rid_val in train_rids:
        train_images.append(rid_to_mri_map[rid_val])

    test_images = []
    for rid_val in test_rids:
        test_images.append(rid_to_mri_map[rid_val])

    # Create the output dictionary
    # In MATLAB, cells of arrays were used. Here, lists of arrays.
    # If images are multi-dimensional, they remain as such within the list.
    D = {
        'TS': train_images,  # List of MRI image arrays
        'TS_labels': np.array(train_labels),
        'TS_Rid': np.array(train_rids),
        'VS': test_images,   # List of MRI image arrays
        'VS_labels': np.array(test_labels),
        'VS_Rid': np.array(test_rids)
    }
    return D

if __name__ == '__main__':
    # Example Usage
    # Dummy MRI data (e.g., paths to files or actual image arrays)
    # For simplicity, let's use placeholder strings for MRI data
    num_subjects = 20
    # Represent MRI data as simple placeholders for this example
    # In reality, these would be NumPy arrays or paths from load_nifti
    example_mri_data_list = [f"mri_data_subject_{i}" for i in range(num_subjects)]
    example_ages = np.random.randint(20, 80, num_subjects)
    example_rids = np.arange(1001, 1001 + num_subjects)

    # Shuffle them initially to simulate unsorted data
    p = np.random.permutation(num_subjects)
    example_mri_data_list = [example_mri_data_list[i] for i in p]
    example_ages = example_ages[p]
    example_rids = example_rids[p]


    print("Original RIDs:", example_rids)
    print("Original Ages:", example_ages)

    split_data = data_split(example_mri_data_list, example_ages, example_rids)

    print("\nTraining Set RIDs:", split_data['TS_Rid'])
    print("Training Set Labels (Ages):", split_data['TS_labels'])
    print(f"Number of Training Images: {len(split_data['TS'])}")
    # print("Training Set Images (placeholders):", split_data['TS'])


    print("\nTest Set RIDs:", split_data['VS_Rid'])
    print("Test Set Labels (Ages):", split_data['VS_labels'])
    print(f"Number of Test Images: {len(split_data['VS'])}")
    # print("Test Set Images (placeholders):", split_data['VS'])

    # Verification of sorting and splitting logic
    all_rids_from_split = np.concatenate((split_data['TS_Rid'], split_data['VS_Rid']))
    all_ages_from_split = np.concatenate((split_data['TS_labels'], split_data['VS_labels']))

    # Check if all original RIDs are present
    assert np.all(np.sort(all_rids_from_split) == np.sort(example_rids))
    print("\nAll RIDs accounted for.")

    # Check if ages are sorted within the combined list (approximately, due to split)
    # The full list of ages (sorted_ages in the function) should be what we check against.
    idx_sorted_original = np.argsort(example_ages)
    original_sorted_ages = example_ages[idx_sorted_original]

    # Reconstruct the sorted order from the split
    combined_ages_in_split_order = []
    combined_rids_in_split_order = [] # RIDs corresponding to combined_ages_in_split_order

    temp_train_idx = 0
    temp_test_idx = 0
    for i in range(len(example_rids)):
        if (i + 1) % 4 == 0:
            combined_ages_in_split_order.append(split_data['VS_labels'][temp_test_idx])
            combined_rids_in_split_order.append(split_data['VS_Rid'][temp_test_idx])
            temp_test_idx += 1
        else:
            combined_ages_in_split_order.append(split_data['TS_labels'][temp_train_idx])
            combined_rids_in_split_order.append(split_data['TS_Rid'][temp_train_idx])
            temp_train_idx += 1

    print("Ages sorted by original logic:", original_sorted_ages)
    print("Ages as they appear in the split (should be sorted):", combined_ages_in_split_order)
    assert np.all(combined_ages_in_split_order == original_sorted_ages)
    print("Age sorting maintained correctly through the split.")