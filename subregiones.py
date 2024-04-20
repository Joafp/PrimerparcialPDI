import zipfile
import io
import random
import cv2
import numpy as np
from PIL import Image, ImageFilter
def divide_imagen(image, L):
    # Calcula los límites para la división de la imagen
    intensidad_min = np.min(image)
    intensidad_max = np.max(image)
    rango_intensidad = intensidad_max - intensidad_min
    paso = rango_intensidad / L
    # Inicializa una lista para almacenar las sub-imágenes
    sub_imagenes = []
    # Divide la imagen en sub-imágenes basadas en los límites calculados
    for i in range(L):
        limite_inferior = intensidad_min + i * paso
        limite_superior = intensidad_min + (i + 1) * paso
        sub_imagen = np.where((image >= limite_inferior) & (image < limite_superior), image, 0)
        sub_imagenes.append(sub_imagen)
    return sub_imagenes
def unir_imagenes(subimagenes):
    # Inicializa una imagen en blanco del mismo tamaño que las subimágenes
    imagen_unida = np.zeros_like(subimagenes[0])
    # Suma las subimágenes pixel a pixel
    for subimagen in subimagenes:
        imagen_unida += subimagen
    return imagen_unida
def calculate_pdf(image):
    frequencies = np.zeros(256, dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            intensity = image[i, j]
            frequencies[intensity] += 1
    total_pixels = image.shape[0] * image.shape[1]
    pdf = frequencies / total_pixels
    return pdf
def sum_pdf(pdf, lower_limit, upper_limit):
    # Verificar que los límites estén dentro del rango
    lower_limit = max(0, lower_limit)
    upper_limit = min(len(pdf), upper_limit)
    # Sumar los valores de pdf en el rango especificado
    sum_value = np.sum(pdf[lower_limit:upper_limit])
    return sum_value
def srhe(sub_imagenes):
    eq_imagenes=[]
    for sub in sub_imagenes:
        pdf=calculate_pdf(sub)
        X0 = np.min(sub)
        XL_1 = np.max(sub)
        total_sum = sum_pdf(pdf, X0, XL_1)
        alto,ancho=sub.shape
        for i in range(alto):
            for j in range(ancho):
                pixel=sub[i,j]
                px=pdf[pixel]
                sum_pk=total_sum
                t_x=X0+(XL_1-X0)*(0.5*px+sum_pk)
                sub[i,j]=t_x
        eq_imagenes.append(sub)
    return eq_imagenes
def gaussian_filter(size, sigma):
    #Calculo del kernel mediante ecucion dada en el material
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)#La division se hace para normalizar

