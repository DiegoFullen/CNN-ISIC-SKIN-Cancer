import os
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import Utils.evaluation as evaluate

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image

class UseModel:
    
    IMAGES_PATH = "Test/"
    DATASET_PATH = "Datasets/metadata-test.csv"

    def __init__(self, model):
        self.model = model
        self.class_names = None  # Se asigna despues
        
    def predict(self, n_images):
        
        # Cargar CSV
        df = pd.read_csv(self.DATASET_PATH)
        df = df.dropna(subset=["isic_id", "benign_malignant"])
        if df.empty:
            raise ValueError("El DataFrame está vacío después de cargar los datos.")

        selected = df.sample(n=n_images)

        # Preparar datos
        X, y = [], []
        for _, row in selected.iterrows():
            path = os.path.join(self.IMAGES_PATH, row["isic_id"] + ".jpg") 

            if not os.path.exists(path):
                continue  # Saltar si no existe

            img = image.load_img(path, target_size=(256, 256))  
            img_array = image.img_to_array(img) / 255.0
            X.append(img_array)
            y.append(row["benign_malignant"])

        X = np.array(X)

        # Convertir etiquetas a one-hot
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_categorical = tf.keras.utils.to_categorical(y_encoded)

        self.class_names = list(label_encoder.classes_)

        # Predecir
        predictions = self.model.predict(X)

        # Mostrar predicciones
        self.display_predictions(X, predictions, y_categorical, num_images=n_images)

    def display_predictions(self, X_test, predictions, y_test, num_images=30):
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)

        correct = np.sum(predicted_classes[:num_images] == true_classes[:num_images])
        total = min(num_images, len(X_test))
    
        cols = 8
        rows = math.ceil(num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        fig.suptitle(f"Aciertos: {correct}/{total}", fontsize=16)
        
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            ax.axis('off')

            if i < num_images:
                ax.imshow(X_test[i])
                pred = self.class_names[predicted_classes[i]]
                real = self.class_names[true_classes[i]]
                color = "green" if predicted_classes[i] == true_classes[i] else "red"
                ax.set_title(f"Pred: {pred}\nReal: {real}", color=color, fontsize=9)
            else:
                ax.set_visible(False)
                
        plt.subplots_adjust(hspace=0.5)  # Espacio vertical entre filas
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("Graphics/predictions.png")
        plt.show()