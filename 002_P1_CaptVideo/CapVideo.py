# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 19:12:38 2025

@author: taver
"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
##fourcc = cv2.VideoWriter_fourcc(*'XVID')
##out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 ##   out.write(frame)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
##out.release()
cv2.destroyAllWindows()