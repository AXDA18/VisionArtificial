# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 21:56:19 2025

@author: taver
"""
import cv2
import numpy as np

# Cargar imagen
img = cv2.imread('imagen1.jpg')
img_copy = img.copy()

# Seleccionar ROI manualmente (ajusta las coordenadas según tu imagen)
x, y, w, h = 100, 100, 200, 200  # (x, y, ancho, alto)
roi = img[y:y+h, x:x+w]

# Crear una máscara que deje solo el ROI y ponga el fondo negro
mask = np.zeros_like(img)
mask[y:y+h, x:x+w] = roi

# Convertir el ROI a escala de grises para detectar esquinas
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Detección de esquinas con Shi-Tomasi
corners = cv2.goodFeaturesToTrack(gray_roi, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = corners.astype(int)

# Dibujar las esquinas detectadas dentro del ROI
for corner in corners:
    cx, cy = corner.ravel()
    cv2.circle(roi, (cx, cy), 3, (0, 255, 0), -1)

# Volver a insertar el ROI modificado en la imagen
img_with_corners = img.copy()
img_with_corners[y:y+h, x:x+w] = roi

# Mostrar resultados
cv2.imshow('ROI con esquinas', roi)
cv2.imshow('Imagen con solo el ROI (fondo negro)', mask)
cv2.imshow('Imagen con ROI y esquinas detectadas', img_with_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()
