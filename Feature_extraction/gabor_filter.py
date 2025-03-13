from skimage.filters import gabor
import cv2
import numpy as np
def extract_gabor_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    features = []
    frequencies = [0.1, 0.3, 0.5]  # Example frequencies
    for freq in frequencies:
        for theta in [0, np.pi/4, np.pi/2]:  # Orientations
            filt_real, _ = gabor(image, frequency=freq, theta=theta)
            features.extend([np.mean(filt_real), np.std(filt_real)])
    return np.array(features)  # e.g., 18 features (3 freqs * 3 orients * 2 stats)