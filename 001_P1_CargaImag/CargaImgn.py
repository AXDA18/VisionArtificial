# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 18:57:35 2025

@author: taver
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('AOI.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()