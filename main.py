import zipfile
from PIL import Image
import io
import random
from CLAHE import clahe_imagen
from ecualizacion_de_histograma import ecualizacion
from Metricas import calculate_ambe,calculate_contrast,calculate_entropy,calculate_psnr
from subregiones import gaussian_filter,srhe,divide_imagen,unir_imagenes
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
                gaussian_kernel = gaussian_filter(3,0.26)
                smoothed_image = cv2.filter2D(image_gray, -1, gaussian_kernel)
                sub=divide_imagen(smoothed_image,256)
                eq=srhe(sub)
                imagen_unida=unir_imagenes(eq)
                cv2.namedWindow('Imagen Original', cv2.WINDOW_NORMAL)
                cv2.imshow('Imagen Original', imagen_unida)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ruta_guardado2 = 'imagen_original.jpg'
                ruta_guardado = 'imagen_unida.jpg'
                cv2.imwrite(ruta_guardado, imagen_unida)
                cv2.imwrite(ruta_guardado2, image_gray)
                ambe = calculate_ambe(image_gray, imagen_unida)
                psnr = calculate_psnr(image_gray, imagen_unida)
                entropy_image1 = calculate_entropy(imagen_unida)
                contrast_image1 = calculate_contrast(imagen_unida)
                print(f"AMBE para imagen con subregiones: {ambe}")
                print(f"PSNR para imagen con subregiones: {psnr}")
                print(f"Entropía (Imagen subregiones): {entropy_image1}")
                print(f"Contraste (Imagen subregiones): {contrast_image1}")
        else:
            print(f"No se encontraron imágenes en la carpeta '{nombre_carpeta}'.")
    else:
        print("No se encontraron carpetas en el archivo ZIP.")
