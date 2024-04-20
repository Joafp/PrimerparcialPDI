import cv2
import numpy as np
from Metricas import calculate_ambe,calculate_contrast,calculate_entropy,calculate_psnr
def clahe_imagen(image_gray):
    if image_gray is None:
        print("Error: la imagen no se pudo cargar. Verifica la ruta del archivo.")
    else:
        # Crear el objeto CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # Aplicar CLAHE a la imagen en escala de grises
        image_clahe = clahe.apply(image_gray)
        ruta_guardado = 'imagen_clahe.jpg'
        cv2.imwrite(ruta_guardado, image_clahe)
        cv2.namedWindow('Imagen Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('CLAHE', cv2.WINDOW_NORMAL)
        cv2.imshow('Imagen Original', image_gray)
        cv2.imshow('CLAHE', image_clahe)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ambe = calculate_ambe(image_gray, image_clahe)
        psnr = calculate_psnr(image_gray, image_clahe)
        entropy_image1 = calculate_entropy(image_clahe)
        contrast_image1 = calculate_contrast(image_clahe)
        print(f"AMBE para imagen clahe: {ambe}")
        print(f"PSNR para imagen clahe: {psnr}")
        print(f"Entropía (Imagen clahe): {entropy_image1}")
        print(f"Contraste (Imagen clahe): {contrast_image1}")
