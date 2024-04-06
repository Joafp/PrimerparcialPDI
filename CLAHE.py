import cv2
import numpy as np
# Verificar si la imagen se ha cargado correctamente
def clahe_imagen(image_gray):
    if image_gray is None:
        print("Error: la imagen no se pudo cargar. Verifica la ruta del archivo.")
    else:
        # Ecualizar el histograma de la imagen de manera estándar
        image_equalized = cv2.equalizeHist(image_gray)

        # Crear el objeto CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # Aplicar CLAHE a la imagen en escala de grises
        image_clahe = clahe.apply(image_gray)
        cv2.namedWindow('Imagen Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('CLAHE', cv2.WINDOW_NORMAL)
        # Mostrar la imagen original, la imagen con histograma ecualizado y la imagen con CLAHE
        cv2.imshow('Imagen Original', image_gray)
        cv2.imshow('CLAHE', image_clahe)

        # Esperar a que el usuario presione una tecla y cerrar todas las ventanas
        cv2.waitKey(0)
        cv2.destroyAllWindows()