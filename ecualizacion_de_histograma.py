import cv2
import numpy as np
from Metricas import calculate_ambe,calculate_contrast,calculate_entropy,calculate_psnr
def ecualizacion(image_gray):
    if image_gray is None:
        print("Error: la imagen no se pudo cargar. Verifica la ruta del archivo.")
    else:
        # Ecualizar el histograma de la imagen
        image_equalized = cv2.equalizeHist(image_gray)
        # Mostrar la imagen original y la imagen ecualizada
        cv2.namedWindow('Imagen Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Imagen Ecualizada', cv2.WINDOW_NORMAL)
        cv2.imshow('Imagen Original', image_gray)
        cv2.imshow('Imagen Ecualizada', image_equalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ambe = calculate_ambe(image_gray, image_equalized)
        psnr = calculate_psnr(image_gray, image_equalized)
        entropy_image1 = calculate_entropy(image_equalized)
        contrast_image1 = calculate_contrast(image_equalized)
        print(f"AMBE para imagen ecualizada: {ambe}")
        print(f"PSNR para imagen ecualizada: {psnr}")
        print(f"Entropía (Imagen ecualizada): {entropy_image1}")
        print(f"Contraste (Imagen ecualizada): {contrast_image1}")