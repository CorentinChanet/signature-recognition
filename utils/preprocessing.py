import cv2
import os
import numpy as np

def load_image(folder_path : str, image_name : str):
    '''Docstring'''
    folder_path = os.path.abspath(folder_path)
    image_path = os.path.join(folder_path, image_name)
    image = cv2.imread(image_path)

    return image


def enhance_image(image : np.ndarray) -> np.ndarray:
    '''Docstring'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return thresh