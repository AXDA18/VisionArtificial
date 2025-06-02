# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 19:29:30 2025

@author: taver
"""
import cv2
import numpy as np

# Cargar imagen principal
img = cv2.imread('imagen.jpg')
img_copy = img.copy()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Cargar template 1
template1 = cv2.imread('template1.jpg')
template1_gray = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
w1, h1 = template1.shape[1], template1.shape[0]

# Cargar template 2
template2 = cv2.imread('template2.jpg')
template2_gray = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
w2, h2 = template2.shape[1], template2.shape[0]

# Umbral de coincidencia
umbral = 0.85

# ---------- Coincidencias para Template 1 ----------
res1 = cv2.matchTemplate(gray_img, template1_gray, cv2.TM_CCOEFF_NORMED)
loc1 = np.where(res1 >= umbral)

for pt in zip(*loc1[::-1]):
    cv2.rectangle(img_copy, pt, (pt[0] + w1, pt[1] + h1), (0, 255, 0), 2)  # Verde

# ---------- Coincidencias para Template 2 ----------
res2 = cv2.matchTemplate(gray_img, template2_gray, cv2.TM_CCOEFF_NORMED)
loc2 = np.where(res2 >= umbral)

for pt in zip(*loc2[::-1]):
    cv2.rectangle(img_copy, pt, (pt[0] + w2, pt[1] + h2), (255, 0, 0), 2)  # Azul

# Mostrar resultado
cv2.imshow('Coincidencias encontradas', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()