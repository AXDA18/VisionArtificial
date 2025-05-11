# -*- coding: utf-8 -*-
"""
Created on Sun May 11 11:03:03 2025
@author: taver
"""

import cv2
import matplotlib.pyplot as plt

# Leer imagen original
img = cv2.imread('bookpage.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # para visualizar con matplotlib
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbralización simple (a color — no común, solo como referencia)
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

# Umbralización simple (escala de grises)
retval2, threshold2 = cv2.threshold(grayscaled, 12, 255, cv2.THRESH_BINARY)

# Umbralización adaptativa gaussiana
gaus = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 115, 1)

# Umbralización con Otsu
retval3, otsu = cv2.threshold(grayscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Crear una figura grande
fig, axs = plt.subplots(5, 2, figsize=(10, 15))

# Mostrar imagen original + histograma
axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title("Imagen original")
axs[0, 0].axis('off')
axs[0, 1].hist(grayscaled.ravel(), 256, [0, 256])
axs[0, 1].set_title("Histograma (escala de grises)")

# Imagen umbralizada simple
axs[1, 0].imshow(threshold2, cmap='gray')
axs[1, 0].set_title("Umbral fijo (12)")
axs[1, 0].axis('off')
axs[1, 1].hist(threshold2.ravel(), 256, [0, 256])
axs[1, 1].set_title("Histograma (umbral fijo)")

# Imagen umbral adaptativo gaussiano
axs[2, 0].imshow(gaus, cmap='gray')
axs[2, 0].set_title("Umbral adaptativo (gauss)")
axs[2, 0].axis('off')
axs[2, 1].hist(gaus.ravel(), 256, [0, 256])
axs[2, 1].set_title("Histograma (adaptativo)")

# Imagen Otsu
axs[3, 0].imshow(otsu, cmap='gray')
axs[3, 0].set_title("Umbral con Otsu")
axs[3, 0].axis('off')
axs[3, 1].hist(otsu.ravel(), 256, [0, 256])
axs[3, 1].set_title("Histograma (Otsu)")

# Imagen en escala de grises
axs[4, 0].imshow(grayscaled, cmap='gray')
axs[4, 0].set_title("Imagen en escala de grises")
axs[4, 0].axis('off')
axs[4, 1].axis('off')  # sin histograma adicional aquí

plt.tight_layout()
plt.show()
