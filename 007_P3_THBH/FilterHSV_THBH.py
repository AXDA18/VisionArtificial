# -*- coding: utf-8 -*-
"""
Created on Fri May 30 19:26:11 2025

@author: taver
"""


import cv2
import numpy as np

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Kernel estructurante
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

def limpiar_mascara(mask):
    # Aplica suavizado + apertura + cierre
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
    apertura = cv2.morphologyEx(mask_blur, cv2.MORPH_OPEN, kernel)
    cierre = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)
    return cierre

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rangos para rojo (dos bandas)
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

    # Aplicar limpieza a las máscaras
    mask_red_clean = limpiar_mascara(mask_red)
    mask_green_clean = limpiar_mascara(mask_green)
    mask_blue_clean = limpiar_mascara(mask_blue)

    # Aplicar máscaras (sin filtro y con filtro)
    red_raw = cv2.bitwise_and(frame, frame, mask=mask_red)
    red_clean = cv2.bitwise_and(frame, frame, mask=mask_red_clean)

    green_raw = cv2.bitwise_and(frame, frame, mask=mask_green)
    green_clean = cv2.bitwise_and(frame, frame, mask=mask_green_clean)

    blue_raw = cv2.bitwise_and(frame, frame, mask=mask_blue)
    blue_clean = cv2.bitwise_and(frame, frame, mask=mask_blue_clean)

    # Comparativas por color (lado a lado)
    comp_red = np.hstack((red_raw, red_clean))
    comp_green = np.hstack((green_raw, green_clean))
    comp_blue = np.hstack((blue_raw, blue_clean))

    # Mostrar comparativas finales
    cv2.imshow('Comparativa Rojo (Izq: sin filtro | Der: con filtro)', comp_red)
    cv2.imshow('Comparativa Verde (Izq: sin filtro | Der: con filtro)', comp_green)
    cv2.imshow('Comparativa Azul (Izq: sin filtro | Der: con filtro)', comp_blue)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
