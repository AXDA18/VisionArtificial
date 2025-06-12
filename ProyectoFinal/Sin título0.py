# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 12:36:57 2025

@author: taver
"""
import matplotlib.pyplot as plt
from preparar_datos import cargar_datos

X, y = cargar_datos(max_images=10)

for i in range(5):
    plt.imshow(y[i].squeeze(), cmap='gray')
    plt.title(f"Mascara {i}")
    plt.axis('off')
    plt.show()