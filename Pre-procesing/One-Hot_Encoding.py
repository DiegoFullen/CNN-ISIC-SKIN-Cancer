import pandas as pd
import numpy as np
import tensorflow as tf
keras = tf.keras

# Cargar datos
df = pd.read_csv("metadata-training.csv")  # Reemplaza con tu archivo

# Convertir la columna 'benign_malignant' a códigos numéricos
labels = df["benign_malignant"].astype('category').cat.codes

# Aplicar one-hot encoding
labels_one_hot = keras.utils.to_categorical(labels)

# Mostrar el resultado
#print(labels_one_hot)

# Obtener los primeros 10 valores originales y su representación One-Hot
for i in range(10):
    print(f"Etiqueta original: {df['benign_malignant'].iloc[i]} → One-Hot: {labels_one_hot[i]}")
