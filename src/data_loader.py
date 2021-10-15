import numpy as np
import os
import cv2

def load_image(image_path : str) -> np.ndarray:
    '''Docstring'''
    image_path = os.path.abspath(image_path)
    image = cv2.imread(image_path)

    return image