import cv2
import numpy as np

def mejorar_imagen_con_ecualizacion_en_subregiones(imagen, num_regiones=4):
    alto, ancho = imagen.shape[:2]
    alto_subregion = alto // num_regiones
    ancho_subregion = ancho // num_regiones
    imagen_mejorada = np.zeros_like(imagen)
    for y in range(num_regiones):
        for x in range(num_regiones):
            y_inicio = y * alto_subregion
            y_fin = (y + 1) * alto_subregion
            x_inicio = x * ancho_subregion
            x_fin = (x + 1) * ancho_subregion
            subregion = imagen[y_inicio:y_fin, x_inicio:x_fin]
            subregion_ecualizada = cv2.equalizeHist(subregion)
            imagen_mejorada[y_inicio:y_fin, x_inicio:x_fin] = subregion_ecualizada

    return imagen_mejorada
