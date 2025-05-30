import numpy as np
import cv2 # OpenCV for image resizing
# Ensure TensorFlow is installed: pip install tensorflow
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import time

def extract_cnn_features_non_overlapping_triplets(image_volumes, cnn_model_base, target_layer_name):
    """
    Extracts features from MRI image volumes using non-overlapping triplets and a pre-trained CNN.

    Args:
        image_volumes (list): A list of 3D NumPy arrays (MRI volumes).
                              Each volume is expected to be (height, width, num_slices).
        cnn_model_base (tf.keras.Model): The base pre-trained CNN model (e.g., VGG16).
        target_layer_name (str): The name of the layer from which to extract features.

    Returns:
        list: A list of 2D NumPy arrays. Each array contains features for one patient/volume,
              where rows are features from triplets and columns are feature dimensions.
    """
    # Create a new model that outputs the features from the target layer
    feature_extractor_model = Model(inputs=cnn_model_base.input,
                                    outputs=cnn_model_base.get_layer(target_layer_name).output)

    all_patient_features = []

    for j, mri_vol in enumerate(image_volumes):
        tic = time.time()
        # mri_vol is expected to be (height, width, num_slices)
        # Keras VGG16 expects input shape (height, width, 3) where height, width >= 32
        # It also expects pixel values to be in a certain range, handled by preprocess_input.

        # Standard VGG16 input size (can be other sizes, but min 32x32)
        input_size = (cnn_model_base.input_shape[1], cnn_model_base.input_shape[2]) # e.g. (224, 224)

        num_slices = mri_vol.shape[2]
        total_triplets = num_slices // 3  # Integer division

        patient_triplet_features = []

        current_slice_idx = 0
        for k in range(total_triplets):
            # Create a triplet: stack 3 consecutive slices
            # Slices are mri_vol[:, :, slice_idx]
            s1 = mri_vol[:, :, current_slice_idx]
            s2 = mri_vol[:, :, current_slice_idx + 1]
            s3 = mri_vol[:, :, current_slice_idx + 2]

            # Combine into a 3-channel image for VGG input
            # Need to ensure slices are grayscale (2D)
            # If slices have depth (e.g. color channel already), this needs adjustment
            triplet_img = np.stack((s1, s2, s3), axis=-1) # Creates (height, width, 3)

            # Resize to VGG input size
            # cv2.resize expects (width, height) for dsize
            resized_triplet = cv2.resize(triplet_img.astype(np.float32),
                                         (input_size[1], input_size[0]),
                                         interpolation=cv2.INTER_LINEAR)

            # Ensure it's 3 channels, VGG might complain if it gets (224,224) instead of (224,224,3)
            if resized_triplet.ndim == 2: # Should not happen with np.stack above
                 resized_triplet = np.stack([resized_triplet]*3, axis=-1)

            # Add batch dimension and preprocess for VGG
            input_batch = np.expand_dims(resized_triplet, axis=0)
            preprocessed_batch = preprocess_input(input_batch) # Uses VGG16-specific preprocessing

            # Get features
            features = feature_extractor_model.predict(preprocessed_batch, verbose=0)
            patient_triplet_features.append(features.flatten())

            current_slice_idx += 3

        if patient_triplet_features:
            all_patient_features.append(np.array(patient_triplet_features))
        else:
            all_patient_features.append(np.array([])) # Handle cases with <3 slices

        toc = time.time()
        print(f"Finished feature extraction for patient #{j+1} ({toc - tic:.2f} seconds)!")

    return all_patient_features


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Load a pre-trained VGG16 model
    #    `include_top=False` removes the final classification layers.
    vgg_model_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # vgg_model_base.summary() # To see layer names

    # 2. Choose a target layer for feature extraction.
    #    The MATLAB code mentions `res(16).x`. In VGG16, this might correspond to
    #    one of the later convolutional blocks or the final pooling layer.
    #    Common choices: 'block5_pool', 'block4_pool', or even flatten of last conv layer.
    #    Let's use 'block5_pool' as an example. Its output is (batch, 7, 7, 512) for 224x224 input.
    #    If the original MATLAB code's `res(16).x` yielded a [9216] vector, it might have been
    #    from a flattened layer like 'fc1' or 'fc2' if `include_top` was true, or a flattened
    #    convolutional layer output.
    #    The MATLAB `FM_16(end + 1,:) = CNN_repn_16(:);` suggests a flattening.
    #    VGG16 output from `block5_pool` for a 224x224 input is (7, 7, 512). Flattened, this is 25088.
    #    If the desired feature vector size is 9216, you might need to check which MatConvNet VGG layer
    #    `res(16).x` corresponds to and find its equivalent in Keras VGG16, or use a different model/layer.
    #    For now, we'll proceed with 'block5_pool' and flatten its output.
    TARGET_LAYER = 'block5_pool' # Example: (None, 7, 7, 512) for VGG16 224x224 input

    # 3. Prepare dummy image volumes (replace with actual loaded MRI data)
    #    Each volume: (height, width, num_slices)
    #    The height/width of slices can be anything, they will be resized.
    #    Number of slices should be >= 3 for the triplet logic.
    dummy_image_volumes = [
        np.random.rand(128, 128, 40).astype(np.float32) * 255,  # Patient 1, 40 slices -> 13 triplets
        np.random.rand(100, 100, 30).astype(np.float32) * 255   # Patient 2, 30 slices -> 10 triplets
    ]
    print(f"Input image volume 0 shape: {dummy_image_volumes[0].shape}")

    # 4. Extract features
    extracted_features_list = extract_cnn_features_non_overlapping_triplets(
        dummy_image_volumes,
        vgg_model_base,
        TARGET_LAYER
    )

    # `extracted_features_list` contains one array per patient.
    # Each array has shape (num_triplets_for_patient, feature_dimension)
    for i, patient_feats in enumerate(extracted_features_list):
        print(f"Features for patient {i+1}: shape {patient_feats.shape}")
        if patient_feats.shape[0] > 0:
            # Example: (13, 25088) if block5_pool output is (7,7,512) and flattened
            print(f"  Number of triplets: {patient_feats.shape[0]}")
            print(f"  Feature dimension per triplet: {patient_feats.shape[1]}")

    # Note on MatConvNet's res(16).x :
    # If you know the exact layer type and configuration from MatConvNet's VGG-VD-16 `res(16).x`
    # (e.g., 'conv5_3', 'pool5', 'fc6'), you can find the corresponding Keras layer name
    # from vgg_model_base.summary() and use that for TARGET_LAYER.
    # The size [9216] mentioned in comments (e.g., VGG-VD_IXI(TS)_[9216]) is specific.
    # For VGG16, common feature sizes are:
    #   - block5_pool flattened: 7*7*512 = 25088 (for 224x224 input)
    #   - fc1 (if include_top=True, then remove classifier): 4096
    #   - fc2 (if include_top=True, then remove classifier): 4096
    # To get 9216, it might be a custom concatenation or a specific layer from a different VGG variant
    # or a different input size to the VGG model that results in such a feature map size.
    # If the MATLAB code used `imagenet-vgg-verydeep-16.mat` and `res(16).x` refers to the output
    # of the 16th layer in its sequence (which could be 'conv3_3' or similar if counting only conv/pool),
    # you'd pick that layer.
    # For example, if res(16).x was the output of 'block3_conv3' (16th layer in MatConvNet's typical VGG structure counting),
    # its Keras equivalent is 'block3_conv3'. Output for 224x224: (None, 56, 56, 256). Flattened: 56*56*256 = 802816.
    # This shows the importance of correctly mapping the MatConvNet layer to the Keras layer.
    # The comment `FM_16(end + 1,:) = CNN_repn_16(:);` where CNN_repn_16 is `res(16).x`
    # implies that the 16th result from `vl_simplenn(net,T{k});` was used.
    # In MatConvNet, `vl_simplenn` returns results for all layers. `res(16)` would be the output of the 16th layer.
    # You need to identify which layer that is in the VGG-VD-16 architecture used by MatConvNet and find its Keras counterpart.
    # If 'imagenet-vgg-verydeep-16.mat' from MatConvNet means layers up to 'pool5', then 'pool5' is 'block5_pool'.
    # If it refers to deeper layers like 'fc6', 'fc7' (MatConvNet names), these are 'fc1', 'fc2' in Keras if `include_top=True`.
    # The target size 9216 is not a standard direct output of Keras VGG16 layers.
    # It could be that `res(16).x` was from a different network, or a specific layer that with a certain input size yields 9216 features after flattening.
    # Or, it could be a typo and it's 4096*2 + something, or 3*3*1024 etc.
    # Given the context of `VGG-VD_IXI(TS)_[9216]`, it is highly likely this 9216 is a specific desired feature dimension.
    # One common feature extraction from VGG16 'fc1' (or 'fc6' in some notations) gives 4096 features.
    # 'fc2' (or 'fc7') also gives 4096. Perhaps it was a concatenation of features or a specific intermediate layer.
    # For this example, I've used 'block5_pool'. Adjust `TARGET_LAYER` and potentially the model based on
    # what `res(16).x` truly represented.