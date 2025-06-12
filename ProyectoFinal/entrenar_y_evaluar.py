# -*- coding: utf-8 -*
"""
Created on Thu Jun 12 07:38:38 2025

@author: taver
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model

# Parámetros
IMG_SIZE = 128
NUM_IMAGES = 200
DATASET_DIR = "data/scene_parse_150"

def cargar_datos():
    images = []
    masks = []

    for i in range(NUM_IMAGES):
        img_path = os.path.join(DATASET_DIR, f"img_{i}.png")
        mask_path = os.path.join(DATASET_DIR, f"mask_{i}.png")

        img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
        mask = Image.open(mask_path).resize((IMG_SIZE, IMG_SIZE)).convert("L")

        img = np.array(img) / 255.0
        mask = np.array(mask) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

def unet_model(input_size=(128, 128, 3)):
    inputs = tf.keras.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.UpSampling2D()(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D()(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D()(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Cargar datos
X, y = cargar_datos()

# Crear modelo
model = unet_model()

# Entrenar
history = model.fit(
    X, y,
    validation_split=0.1,
    epochs=10,
    batch_size=8
)

# Graficar pérdida
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Mostrar predicciones
preds = model.predict(X[:5])

for i in range(5):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(X[i])
    plt.title("Imagen")

    plt.subplot(1, 3, 2)
    plt.imshow(y[i].squeeze(), cmap='gray')
    plt.title("Máscara real")

    plt.subplot(1, 3, 3)
    pred = (preds[i].squeeze() > 0.5).astype(np.uint8)
    plt.imshow(pred, cmap='gray')
    plt.title("Predicción")

    plt.tight_layout()
    plt.show()
