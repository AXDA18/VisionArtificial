# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:28:23 2025

@author: taver
"""
# -----------------------------------------
# PROYECTO: Clasificación de ropa con CNN
# Dataset: Fashion MNIST (Hugging Face)
# Autor: [taver]
# -----------------------------------------

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -----------------------------
# 1. Cargar dataset desde Hugging Face
# -----------------------------
dataset = load_dataset("fashion_mnist")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# -----------------------------
# 2. Visualizar algunas imágenes
# -----------------------------
def show_images():
    images = dataset['train']['image']
    labels = dataset['train']['label']

    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_images()

# -----------------------------
# 3. Preprocesamiento
# -----------------------------
def convert_to_numpy(split):
    images = np.array([np.array(img) for img in dataset[split]['image']])
    labels = np.array(dataset[split]['label'])
    return images, labels

x_train, y_train = convert_to_numpy('train')
x_test, y_test = convert_to_numpy('test')

# Normalización y reshape
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., np.newaxis]  # (28, 28, 1)
x_test = x_test[..., np.newaxis]

print("x_train shape:", x_train.shape)

# -----------------------------
# 4. Definir modelo CNN
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------
# 5. Entrenar modelo
# -----------------------------
history = model.fit(x_train, y_train, epochs=10,
                    validation_split=0.2, batch_size=64)

# -----------------------------
# 6. Evaluar el modelo
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nPrecisión en datos de prueba: {test_acc:.4f}')

# -----------------------------
# 7. Predecir y mostrar resultados (10 imágenes)
# -----------------------------

# Hacer predicciones del modelo sobre x_test
predictions = model.predict(x_test)

# Mostrar imágenes con etiquetas reales y predichas
def show_predictions(start_index=0, cantidad=10):
    plt.figure(figsize=(15, 5))
    for i in range(cantidad):
        idx = start_index + i
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        true_label = class_names[y_test[idx]]
        pred_label = class_names[np.argmax(predictions[idx])]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"Real: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Ejecutar visualización (ejemplo: primeras 10 imágenes)
show_predictions(0, 10)