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

# -*- coding: utf-8 -*-
"""
PROYECTO: Clasificación de ropa con CNN (Fashion MNIST)
Mejorado con aumento de datos y validación manual
"""

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Cargar dataset desde Hugging Face
dataset = load_dataset("fashion_mnist")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 2. Visualizar algunas imágenes
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

# 3. Preprocesamiento
def convert_to_numpy(split):
    images = np.array([np.array(img) for img in dataset[split]['image']])
    labels = np.array(dataset[split]['label'])
    return images, labels

x_train_full, y_train_full = convert_to_numpy('train')
x_test, y_test = convert_to_numpy('test')

# Normalización
x_train_full = x_train_full / 255.0
x_test = x_test / 255.0

# Añadir canal (28,28,1)
x_train_full = x_train_full[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# División manual en entrenamiento y validación (80/20)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=42)

# 4. Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# 5. Definir modelo CNN
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

# 6. Callbacks para regularización
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# 7. Entrenamiento
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_val, y_val),
    epochs=30,
    callbacks=[early_stop, reduce_lr]
)

# 8. Evaluación
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nPrecisión en datos de prueba: {test_acc:.4f}')

# 9. Predicciones
predictions = model.predict(x_test)

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

show_predictions(0, 10)