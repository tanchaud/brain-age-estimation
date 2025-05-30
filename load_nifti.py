import os
import nibabel as nib
import numpy as np

def load_nifti_slices(nii_dir, nii_list, n, num_slices=120):
    """
    Loads NIFTI images and extracts a specific range of slices.

    Args:
        nii_dir (str): Directory containing NIFTI files.
        nii_list (list): List of NIFTI file names (or file objects).
        n (int): Number of images to load.
        num_slices (int, optional): Number of slices to extract from the
                                    beginning of the 3rd dimension. Defaults to 120.

    Returns:
        list: A list of 3D NumPy arrays, where each array is an MRI image volume.
    """
    images = []
    for i in range(n):
        file_name = nii_list[i]
        # If nii_list contains file objects from os.listdir that have a .name attribute
        if hasattr(file_name, 'name'):
            file_path = os.path.join(nii_dir, file_name.name)
        # If nii_list contains just strings
        else:
            file_path = os.path.join(nii_dir, file_name)

        try:
            nii_img = nib.load(file_path)
            mri_image_data = nii_img.get_fdata()

            # Ensure it's at least 3D before slicing
            if mri_image_data.ndim >= 3:
                mri_image_sliced = mri_image_data[:, :, :num_slices]
            else:
                # Handle cases with fewer than 3 dimensions if necessary,
                # or raise an error/warning
                print(f"Warning: Image {file_name} has fewer than 3 dimensions. Skipping slicing.")
                mri_image_sliced = mri_image_data

            images.append(mri_image_sliced)
        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}")

    return images

if __name__ == '__main__':
    # Example Usage (adjust paths and file names as needed)
    # Create dummy NIFTI files for testing
    # In a real scenario, these would be your actual NIFTI files.
    if not os.path.exists('dummy_nii_dir'):
        os.makedirs('dummy_nii_dir')

    for i in range(5):
        img_data = np.random.rand(10, 10, 150) * 1000
        img_data = img_data.astype(np.int16) # Common NIFTI data type
        nifti_img = nib.Nifti1Image(img_data, np.eye(4))
        nib.save(nifti_img, os.path.join('dummy_nii_dir', f'test_image_{i}.nii'))

    dummy_nii_dir = 'dummy_nii_dir'
    # Simulating MATLAB's dir command output for .nii files
    # In Python, os.listdir() gives names, glob.glob can filter
    dummy_nii_list = [f for f in os.listdir(dummy_nii_dir) if f.endswith('.nii')]
    num_files_to_load = min(5, len(dummy_nii_list)) # Load up to 5 files

    loaded_images = load_nifti_slices(dummy_nii_dir, dummy_nii_list, num_files_to_load)

    if loaded_images:
        print(f"Successfully loaded {len(loaded_images)} images.")
        print(f"Shape of the first loaded image: {loaded_images[0].shape}")
    else:
        print("No images were loaded.")

    # Clean up dummy files and directory
    # for f in dummy_nii_list:
    #     os.remove(os.path.join(dummy_nii_dir, f))
    # os.rmdir(dummy_nii_dir)