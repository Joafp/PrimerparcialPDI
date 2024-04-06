import cv2
import numpy as np
from skimage.measure import shannon_entropy
def calculate_ambe(image1, image2):
    return abs(np.mean(image1) - np.mean(image2))

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_entropy(image):
    return shannon_entropy(image)

def calculate_contrast(image):
    return np.std(image)
