# -*- coding: utf-8 -*-
"""
Created on Sat May 10 20:49:48 2025

@author: taver
"""

import cv2
import matplotlib.pyplot as plt

# Cargar la imagen
img = cv2.imread('AOI.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para matplotlib

# Dimensiones de la imagen
h, w, _ = img.shape

# Dibujar un rectángulo (ROI visual)
start_point = (int(w*0.35), int(h*0.35))  # punto superior izquierdo
end_point = (int(w*0.7), int(h*0.7))    # punto inferior derecho
color = (255, 0, 0)  # rojo en RGB
thickness = 3
cv2.rectangle(img, start_point, end_point, color, thickness)

# Dibujar un círculo
center = (int(w/2), int(h/2))
cv2.circle(img, center, 30, (0, 255, 0), 2)  # verde

# Escribir texto
cv2.putText(img, 'AOI Machine', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# Extraer ROI real de la imagen
roi = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]

# Mostrar imagen completa y ROI
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img)
axs[0].set_title('Imagen con dibujo y texto')
axs[0].axis('off')

axs[1].imshow(roi)
axs[1].set_title('ROI segmentada')
axs[1].axis('off')

plt.tight_layout()
plt.show()
