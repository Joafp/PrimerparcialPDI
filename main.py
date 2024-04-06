import zipfile
from PIL import Image
import io
import random
from CLAHE import clahe_imagen
from ecualizacion_en_subregiones import mejorar_imagen_con_ecualizacion_en_subregiones
from ecualizacion_de_histograma import ecualizacion
from Metricas import calculate_ambe,calculate_contrast,calculate_entropy,calculate_psnr
import cv2
import numpy as np
archivo_zip = 'Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02.zip'
with zipfile.ZipFile(archivo_zip, 'r') as archivo_zip:
    lista_elementos = archivo_zip.namelist()
    carpetas = [nombre for nombre in lista_elementos if archivo_zip.getinfo(nombre).is_dir()]
    if carpetas:
        nombre_carpeta = random.choice(carpetas)
        imagenes_carpeta = [nombre for nombre in lista_elementos if nombre.startswith(nombre_carpeta) and nombre.endswith('.jpg')]
        if imagenes_carpeta:
            nombre_imagen = random.choice(imagenes_carpeta)
            with archivo_zip.open(nombre_imagen) as archivo_imagen:
                imagen_bytes = archivo_imagen.read()
                # Leer la imagen en escala de grises usando OpenCV
                imagen_np = np.asarray(bytearray(imagen_bytes), dtype=np.uint8)
                image_gray = cv2.imdecode(imagen_np, cv2.IMREAD_GRAYSCALE)   
                clahe_imagen(image_gray)        
                ecualizacion(image_gray)
                imagen_mejorada = mejorar_imagen_con_ecualizacion_en_subregiones(image_gray)
                cv2.namedWindow('Imagen Original', cv2.WINDOW_NORMAL)
                cv2.namedWindow('Imagen Mejorada con Ecualización en Subregiones', cv2.WINDOW_NORMAL)
                cv2.imshow('Imagen Original', image_gray)
                cv2.imshow('Imagen Mejorada con Ecualización en Subregiones', imagen_mejorada)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ambe = calculate_ambe(image_gray, imagen_mejorada)
                psnr = calculate_psnr(image_gray, imagen_mejorada)
                entropy_image1 = calculate_entropy(imagen_mejorada)
                contrast_image1 = calculate_contrast(imagen_mejorada)
                print(f"AMBE para imagen con subregiones: {ambe}")
                print(f"PSNR para imagen con subregiones: {psnr}")
                print(f"Entropía (Imagen subregiones): {entropy_image1}")
                print(f"Contraste (Imagen subregiones): {contrast_image1}")
        else:
            print(f"No se encontraron imágenes en la carpeta '{nombre_carpeta}'.")
    else:
        print("No se encontraron carpetas en el archivo ZIP.")
