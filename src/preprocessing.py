import cv2
import numpy as np

def resize_image(image : np.ndarray, dim) -> np.ndarray:
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def enhance_image(image : np.ndarray, erosion_param=True) -> np.ndarray:
    '''Docstring'''

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    if erosion_param:
        erosion = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel, iterations=1)

        thresh = cv2.threshold(erosion, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return thresh