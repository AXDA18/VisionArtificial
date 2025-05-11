import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imágenes y convertir a escala de grises
img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Redimensionar la segunda imagen al tamaño de la primera
img2_gray_resized = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

# -------- HISTOGRAMA ORIGINAL Y ECUALIZACIÓN --------
hist_orig = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
img_eq = cv2.equalizeHist(img1_gray)
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# -------- OPERACIONES ARITMÉTICAS --------
suma = cv2.add(img1_gray, img2_gray_resized)
resta = cv2.subtract(img1_gray, img2_gray_resized)

# Reducción
reduccion = cv2.resize(img1_gray, (img1_gray.shape[1]//2, img1_gray.shape[0]//2))

# Rotación
(h, w) = img1_gray.shape
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1)
rotada = cv2.warpAffine(img1_gray, M, (w, h))

# -------- VISUALIZACIÓN EN SUBPLOTS --------
fig, axs = plt.subplots(3, 3, figsize=(16, 12))

# Fila 1: Imagen original, histograma original, imagen ecualizada
axs[0, 0].imshow(img1_gray, cmap='gray')
axs[0, 0].set_title("Imagen original")
axs[0, 0].axis('off')

axs[0, 1].plot(hist_orig, color='black')
axs[0, 1].set_title("Histograma original")

axs[0, 2].imshow(img_eq, cmap='gray')
axs[0, 2].set_title("Imagen ecualizada")
axs[0, 2].axis('off')

# Fila 2: Histograma ecualizado, imagen suma, imagen resta
axs[1, 0].plot(hist_eq, color='black')
axs[1, 0].set_title("Histograma ecualizado")

axs[1, 1].imshow(suma, cmap='gray')
axs[1, 1].set_title("Suma")
axs[1, 1].axis('off')

axs[1, 2].imshow(resta, cmap='gray')
axs[1, 2].set_title("Resta")
axs[1, 2].axis('off')

# Fila 3: Imagen reducida, imagen rotada, imagen2 redimensionada
axs[2, 0].imshow(reduccion, cmap='gray')
axs[2, 0].set_title("Reducción de tamaño")
axs[2, 0].axis('off')

axs[2, 1].imshow(rotada, cmap='gray')
axs[2, 1].set_title("Rotación 45°")
axs[2, 1].axis('off')

axs[2, 2].imshow(img2_gray_resized, cmap='gray')
axs[2, 2].set_title("Imagen 2 redimensionada")
axs[2, 2].axis('off')

plt.tight_layout()
plt.show()