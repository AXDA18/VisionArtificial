# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 22:47:07 2025

@author: taver
"""
import cv2
import numpy as np

# Cargar imágenes en escala de grises
img1 = cv2.imread('imagen1.jpg', cv2.IMREAD_GRAYSCALE)  # Imagen original
img2 = cv2.imread('imagen2.jpg', cv2.IMREAD_GRAYSCALE)  # Imagen recortada y rotada

# Verificar que se cargaron correctamente
if img1 is None or img2 is None:
    print("Error al cargar las imágenes.")
    exit()

# Detectar puntos clave y descriptores con ORB
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Verificar descriptores
if des1 is None or des2 is None:
    print("No se detectaron suficientes descriptores.")
    exit()

# Emparejar con BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)

# Filtrar coincidencias buenas con regla de Lowe
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Necesitamos al menos 4 coincidencias buenas para calcular la homografía
if len(good_matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calcular homografía con RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Dibujar solo las coincidencias válidas
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
else:
    print("No se encontraron suficientes coincidencias válidas.")
    result = cv2.drawMatches(img1, kp1, img2, kp2, [], None, flags=2)

# Redimensionar resultado si es necesario
result = cv2.resize(result, (0, 0), fx=0.8, fy=0.6)

cv2.imshow('Coincidencias válidas con RANSAC', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
