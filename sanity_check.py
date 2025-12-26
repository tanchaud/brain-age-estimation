#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 21:03:21 2025

@author: tanchaud
"""
import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("-" * 30)

try:
    print("Checking NumPy...")
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    # Simple NumPy operation
    a = np.array([1, 2, 3])
    print(f"NumPy test array: {a}")
    print("NumPy: OK")
except ImportError as e:
    print(f"Error importing NumPy: {e}")
print("-" * 30)

try:
    print("Checking SciPy...")
    import scipy
    print(f"SciPy version: {scipy.__version__}")
    # Simple SciPy operation (e.g., from constants)
    from scipy import constants
    print(f"SciPy test (speed of light): {constants.c}")
    print("SciPy: OK")
except ImportError as e:
    print(f"Error importing SciPy: {e}")
print("-" * 30)

try:
    print("Checking Matplotlib...")
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
    # Check backend (no actual plotting needed for sanity check)
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print("Matplotlib: OK")
except ImportError as e:
    print(f"Error importing Matplotlib: {e}")
print("-" * 30)

try:
    print("Checking OpenCV (cv2)...")
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
    # You can also check build information for more details if needed
    # print(f"OpenCV build info: {cv2.getBuildInformation()[:200]}...") # Print first 200 chars
    print("OpenCV (cv2): OK")
except ImportError as e:
    print(f"Error importing OpenCV (cv2): {e}")
print("-" * 30)

try:
    print("Checking Scikit-learn (sklearn)...")
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
    # Simple model import test
    from sklearn.svm import SVR
    print("Scikit-learn SVR import: OK")
    print("Scikit-learn: OK")
except ImportError as e:
    print(f"Error importing Scikit-learn: {e}")
print("-" * 30)

try:
    print("Checking Nibabel...")
    import nibabel
    print(f"Nibabel version: {nibabel.__version__}")
    print("Nibabel: OK")
except ImportError as e:
    print(f"Error importing Nibabel: {e}")
print("-" * 30)

try:
    print("Checking TensorFlow...")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    # Check if GPU is available (optional, but good to know)
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"TensorFlow GPU devices available: {gpu_devices}")
    else:
        print("TensorFlow: No GPU devices found, will use CPU.")
    # Simple TensorFlow operation
    hello = tf.constant('Hello from TensorFlow!')
    print(f"TensorFlow constant: {hello.numpy().decode()}")
    print("TensorFlow: OK")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
except Exception as e_tf: # Catch other potential TF errors like CUDA issues
    print(f"Error during TensorFlow check (beyond import): {e_tf}")
print("-" * 30)

print("Sanity check complete.")