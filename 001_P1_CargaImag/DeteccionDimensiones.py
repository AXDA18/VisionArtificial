# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 13:34:03 2025

@author: taver
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('AOI.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

plt.plot([200,225,400],[100,250,225],'c', linewidth=2)
plt.show()

cv2.imwrite('AOIMeasure.png',img)