import cv2
import numpy as np
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