# preparar_datos.py
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:38:06 2025
@author: taver
"""
import numpy as np
import os
from tensorflow.keras.utils import img_to_array, load_img
from sklearn.model_selection import train_test_split

def cargar_datos(ruta_imagenes, ruta_mascaras, tamaño=(128, 128)):
    imagenes = []
    mascaras = []

    # Solo usar imágenes 0, 1, 3 y 4
    nombres_validos = ["0.png", "1.png", "3.png", "4.png"]

    for nombre_archivo in sorted(os.listdir(ruta_imagenes)):
        if nombre_archivo in nombres_validos:
            img = load_img(os.path.join(ruta_imagenes, nombre_archivo), target_size=tamaño)
            msk = load_img(os.path.join(ruta_mascaras, nombre_archivo), color_mode="grayscale", target_size=tamaño)

            img = img_to_array(img) / 255.0
            msk = img_to_array(msk) / 255.0
            msk = (msk > 0.5).astype(np.float32)

            imagenes.append(img)
            mascaras.append(msk)

    imagenes = np.array(imagenes)
    mascaras = np.array(mascaras)

    return train_test_split(imagenes, mascaras, test_size=0.25, random_state=42)
