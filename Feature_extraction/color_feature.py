import cv2
import numpy as np

def extract_color_features(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize if necessary (e.g., to 224x224)
    image = cv2.resize(image, (224, 224))
    # Compute mean and standard deviation
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    return np.array([mean_intensity, std_intensity])