# -*- coding: utf-8 -*-
"""
Created on Fri May 9 11:34:57 2025

@author: taver
"""

import cv2
import numpy as np

# Captura de video desde la cámara (puedes cambiar '0' por el índice de otra cámara o una ruta de video)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rangos para ROJO (en HSV hay dos rangos por el ciclo de 180°)
    lower_red1 = np.array([0, 100, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Verde
    lower_green = np.array([36, 60, 40])
    upper_green = np.array([86, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Azul
    lower_blue = np.array([94, 60, 2])
    upper_blue = np.array([126, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Aplicar las máscaras
    red_result = cv2.bitwise_and(frame, frame, mask=mask_red)
    green_result = cv2.bitwise_and(frame, frame, mask=mask_green)
    blue_result = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # Mostrar resultados
    cv2.imshow('Original', frame)
    cv2.imshow('Filtro Rojo', red_result)
    cv2.imshow('Filtro Verde', green_result)
    cv2.imshow('Filtro Azul', blue_result)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()