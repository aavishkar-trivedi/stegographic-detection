import numpy as np
import cv2

def extract_features(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    variance = np.var(image)
    mean = np.mean(image)

    score = abs(variance - 0.08)

    return {
        "feature_score": float(score),
        "mean": float(mean),
        "variance": float(variance)
    }
