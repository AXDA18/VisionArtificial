# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:18:12 2025

@author: taver
"""

import cv2
import numpy as np

img = cv2.imread('AOI.jpg',cv2.IMREAD_COLOR)

img[25,55] = [255,255,255]
px = img[25,55]
img[100:150,100:150] = [255,255,255]

watch_face = img[37:111,107:194]
img[0:74,0:87] = watch_face
img_grande = cv2.resize(img, (0,0), fx=2, fy=2)

cv2.imshow('Imagen Modificada', img_grande)
cv2.waitKey(0)
cv2.destroyAllWindows()
