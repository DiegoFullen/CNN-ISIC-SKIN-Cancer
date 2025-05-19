import pandas as pd
import os
import cv2  # Para cargar imágenes
import numpy as np
import tensorflow as tf

keras = tf.keras
from sklearn.model_selection import train_test_split

# Ruta de las carpetas (ajústalas a tu caso)
TRAIN_DIR = "Training"
TEST_DIR = "TestSi, era una"
CSV_FILE = "metadata-training.csv"  # Ruta del archivo con etiquetas

# Cargar el CSV
df = pd.read_csv(CSV_FILE)

# Convertir etiquetas de texto a números
df["benign_malignant"] = df["benign_malignant"].astype("category")
df["label_encoded"] = df["benign_malignant"].cat.codes

# Aplicar One-Hot Encoding
labels_one_hot = keras.utils.to_categorical(df["label_encoded"])

# Tamaño de las imágenes (ajústalo si es necesario)
IMG_SIZE = 224  # Usaremos imágenes de 224x224 píxeles

# Función para cargar imágenes
def load_images_from_folder(folder, df):
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = os.path.join(folder, row["isic_id"] + ".jpg")  # Asume que las imágenes son .jpg
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Redimensionar
            img = img / 255.0  # Normalizar
            images.append(img)
            labels.append(labels_one_hot[index])  # Asocia la imagen con su etiqueta
    return np.array(images), np.array(labels)

# Cargar imágenes de entrenamiento y prueba
X_train, y_train = load_images_from_folder(TRAIN_DIR, df)
X_test, y_test = load_images_from_folder(TEST_DIR, df)

print(f"Imágenes de entrenamiento: {X_train.shape}, Etiquetas: {y_train.shape}")
print(f"Imágenes de prueba: {X_test.shape}, Etiquetas: {y_test.shape}")
