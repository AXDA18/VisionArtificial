# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 22:52:43 2025

@author: taver
"""

import cv2

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Crear sustractor de fondo (modelo por defecto)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar sustracción de fondo
    fgmask = fgbg.apply(frame)

    # Mostrar resultados
    cv2.imshow('Video original', frame)
    cv2.imshow('Movimiento detectado', fgmask)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
