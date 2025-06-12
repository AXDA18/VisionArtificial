# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:38:06 2025

@author: taver
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE = 128
DATA_DIR = "./data/scene_parse_150"

def cargar_datos(n=200):
    X, y = [], []
    for i in range(n):
        img_path = os.path.join(DATA_DIR, f"img_{i}.png")
        mask_path = os.path.join(DATA_DIR, f"mask_{i}.png")

        # Cargar imagen y máscara
        img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
        mask = Image.open(mask_path).resize((IMG_SIZE, IMG_SIZE)).convert('L')

        img = np.array(img) / 255.0
        mask = np.array(mask)

        # Normalizar máscara: 0 (fondo), 1 (objeto)
        mask = (mask > 20).astype(np.float32)  # Umbral configurable

        mask = np.expand_dims(mask, axis=-1)

        X.append(img)
        y.append(mask)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Prueba rápida
if __name__ == "__main__":
    X, y = cargar_datos(6)
    for i in range(6):
        plt.figure(figsize=(8, 3))

        plt.subplot(1, 2, 1)
        plt.imshow(X[i])
        plt.title("Imagen")

        plt.subplot(1, 2, 2)
        plt.imshow(y[i].squeeze(), cmap='gray')
        plt.title("Máscara Real")

        plt.show()
