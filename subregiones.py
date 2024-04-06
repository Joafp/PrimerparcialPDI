import numpy as np
import cv2
def gaussian_filter(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)
def sub_regions_histogram_equalization(image, filter_size, sigma):
    gaussian_kernel = gaussian_filter(filter_size, sigma)
    smoothed_image = cv2.filter2D(image, -1, gaussian_kernel)
    sub_images = np.zeros_like(image)
    num_levels = 256
    for level in range(num_levels):
        mask = (image == level)
        sub_images[mask] = smoothed_image[mask]
    equalized_image = np.zeros_like(image)
    for level in range(num_levels):
        sub_image = (sub_images == level)
        equalized_sub_image = cv2.equalizeHist(sub_image.astype(np.uint8)) * level
        equalized_image += equalized_sub_image

    return equalized_image
