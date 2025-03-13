import cv2
import numpy as np
def extract_shape_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    areas = [cv2.contourArea(c) for c in contours] if contours else [0]
    perimeters = [cv2.arcLength(c, True) for c in contours] if contours else [0]
    avg_area = np.mean(areas) if areas else 0
    avg_perimeter = np.mean(perimeters) if perimeters else 0
    return np.array([num_contours, avg_area, avg_perimeter])  # 3 features