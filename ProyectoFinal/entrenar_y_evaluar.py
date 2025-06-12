# entrenar_y_evaluar.py
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:38:38 2025
@author: taver
"""
import matplotlib.pyplot as plt
from preparar_datos import cargar_datos
from modelo_unet import construir_unet, dice_coef
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Rutas (ajusta según tu estructura de carpetas)
ruta_imagenes = "data/imagenes"
ruta_mascaras = "data/mascaras"

# Cargar datos
x_train, x_val, y_train, y_val = cargar_datos(ruta_imagenes, ruta_mascaras)

# Modelo
model = construir_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])

# Entrenamiento
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=2,
    callbacks=[early_stopping]
)

# Graficar pérdidas
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.title("Curva de Pérdida")
plt.legend()
plt.show()

# Predicciones
preds = model.predict(x_val)

# Visualizar resultados
for i in range(len(x_val)):
    pred = preds[i].squeeze()
    pred_bin = (pred > 0.5).astype(np.uint8)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 4, 1); plt.imshow(x_val[i]); plt.title("Imagen")
    plt.subplot(1, 4, 2); plt.imshow(y_val[i].squeeze(), cmap='gray'); plt.title("Máscara Real")
    plt.subplot(1, 4, 3); plt.imshow(pred, cmap='gray'); plt.title("Predicción Raw")
    plt.subplot(1, 4, 4); plt.imshow(pred_bin, cmap='gray'); plt.title("Predicción Binaria")
    plt.show()
