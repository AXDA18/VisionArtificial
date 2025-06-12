# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 07:38:06 2025

@author: taver
"""
import os
import numpy as np
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = 128
DATA_DIR = "data/scene_parse_150"

def guardar_datos_local(n=200):
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset = load_dataset("scene_parse_150", split="train[:{}]".format(n), trust_remote_code=True)

    for i, example in enumerate(dataset):
        img = example['image'].resize((IMG_SIZE, IMG_SIZE))
        mask = example['annotation'].resize((IMG_SIZE, IMG_SIZE))
        img.save(f"{DATA_DIR}/img_{i}.png")
        mask.save(f"{DATA_DIR}/mask_{i}.png")

def cargar_datos_local():
    X, y = [], []
    files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("img_")])
    for f in files:
        idx = f.split('_')[1].split('.')[0]
        img = np.array(Image.open(f"{DATA_DIR}/img_{idx}.png")) / 255.0
        mask = np.array(Image.open(f"{DATA_DIR}/mask_{idx}.png").convert('L')) / 255.0
        X.append(img.astype(np.float32))
        y.append(np.expand_dims(mask.astype(np.float32), axis=-1))
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Solo necesitas ejecutar esto una vez con internet
    guardar_datos_local()
    print("Datos descargados y guardados localmente.")
