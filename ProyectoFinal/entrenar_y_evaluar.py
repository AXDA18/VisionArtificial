# -*- coding: utf-8 -*
"""
Created on Thu Jun 12 07:38:38 2025

@author: taver
"""
from preparar_datos import cargar_datos
from modelo_unet import unet_model
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos
X_train, y_train = cargar_datos(200)

# Crear modelo
model = unet_model()

# Entrenar
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=8
)

# Guardar modelo entrenado
model.save("modelo_entrenado.keras")

# Gráfica del entrenamiento
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.title("Curva de Pérdida")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Visualizar predicciones
preds = model.predict(X_train[:5])

for i in range(5):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(X_train[i])
    plt.title("Imagen")

    plt.subplot(1, 4, 2)
    plt.imshow(y_train[i].squeeze(), cmap='gray')
    plt.title("Máscara Real")

    plt.subplot(1, 4, 3)
    plt.imshow(preds[i].squeeze(), cmap='gray')
    plt.title("Predicción Raw")

    plt.subplot(1, 4, 4)
    pred_bin = (preds[i].squeeze() > 0.5).astype(np.uint8)
    plt.imshow(pred_bin, cmap='gray')
    plt.title("Predicción Binaria")

    plt.tight_layout()
    plt.show()
