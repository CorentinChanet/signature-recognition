import cv2
import os
import numpy as np

def load_image(image_path : str) -> np.ndarray:
    '''Docstring'''
    image_path = os.path.abspath(image_path)
    image = cv2.imread(image_path)

    return image


def enhance_image(image : np.ndarray) -> np.ndarray:
    '''Docstring'''

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel, iterations=1)

    thresh = cv2.threshold(erosion, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return thresh