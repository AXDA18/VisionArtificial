# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 15:46:32 2025

@author: taver
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer imagen y convertir a escala de grises
img = cv2.imread('AOI.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar filtro Laplaciano
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Aplicar Sobel X
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobelx = np.uint8(np.absolute(sobelx))

# Aplicar Sobel Y
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobely = np.uint8(np.absolute(sobely))

# Aplicar filtro Canny
canny = cv2.Canny(gray, 100, 200)

# Mostrar resultados en una figura
titles = ['Original', 'Laplaciano', 'Sobel X', 'Sobel Y', 'Canny']
images = [gray, laplacian, sobelx, sobely, canny]

plt.figure(figsize=(10, 8))

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
