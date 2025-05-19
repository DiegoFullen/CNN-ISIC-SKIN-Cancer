import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import config.settings as settings

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from Utils.evaluation import evaluate # Clase custom

from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

keras = tf.keras

TEST_CSV_FILE = "datasets/metadata-test copy.csv"
TRAIN_CSV_FILE = "datasets/metadata-training copy.csv"
TRAIN_DIR = "Training"
TEST_DIR = "Test"
MODEL_DIR = "Model"
    
#Hyperparámetros
IMG_SIZE = 256  # Aumentado ligeramente para capturar más detalles
CHANNEL_SIZE = 3  # Número de canales de la imagen (RGB)

# CNN Hyperparameters
EPOCHS = 10  # Aumentado para dar más tiempo al entrenamiento
BATCH_SIZE = 16  # Reducido para mejor generalización
NEURONS_DENSE = 512  # Número de neuronas en la capa densa
NEURONS_DENSE2 = 256  # Número de neuronas en la segunda capa densa
DROPOUT_RATE = 0.5  # Ajustado para mejor balance
DROPOUT_RATE2 = 0.3
L2_LAMBDA = 0.0001  # Regularización L2 para reducir sobreajuste

# Mejorado el diseño de filtros
CONV_FILTER = {
    "conv1": 32,
    "conv2": 64,
    "conv3": 128,
    "conv4": 256,
    "conv5": 512,  # Nueva capa convolucional
}

CLASS_WEIGHT_BENIGN = 1.0  
CLASS_WEIGHT_MALIGNANT = 4.0  
class_weight_dict = {0: CLASS_WEIGHT_BENIGN, 1: CLASS_WEIGHT_MALIGNANT}

# Separacion de datos por bloques
BLOCK_SIZE = 500  # Imágenes por bloque de entreno

class CNN:
    def __init__(self):
        self.train_df = pd.read_csv(TRAIN_CSV_FILE)
        self.test_df = pd.read_csv(TEST_CSV_FILE)
        self.settings = settings.Settings()
         # Inicializar atributos para guardar información
        self.accuracy = 0.0
        self.loss = 0.0
        self.f1 = 0.0
        self.recall = 0.0
        self.matrix_confusion = {}
    
    def train_model(self):
        # Preprocesar los datos
        self.preprocess_data()

        # Crear el modelo
        self.model = self.create_cnn_model()

        # Validar el modelo
        self.validate_model()

        # Compilar el modelo
        self.compile_model()

        # Guardar el modelo entrenado
        self.save_model()
        
        # Registrar historial de entrenamiento
        self.save_settings()
        
    def preprocess_data(self):
        # Preprocesar ambos dataframes
        for df in [self.train_df, self.test_df]:
            df["benign_malignant"] = df["benign_malignant"].astype("category")
            df["label_encoded"] = df["benign_malignant"].cat.codes

        # Obtener el número de clases
        self.NUM_CLASSES = len(self.train_df["benign_malignant"].cat.categories)
        print(f"Número de clases detectadas: {self.NUM_CLASSES}")
        print(f"Clases: {self.train_df['benign_malignant'].cat.categories.tolist()}")

        # Calcular las proporciones actuales
        class_counts = self.train_df['benign_malignant'].value_counts()
        print("Distribución de clases:")
        print(class_counts)

        # Calcular dinámicamente los pesos de clase
        class_weights = compute_class_weight('balanced', classes=np.unique(self.train_df["label_encoded"]), y = self.train_df["label_encoded"])
        class_weight_dict = {i: weight for i, weight in zip(np.unique(self.train_df["label_encoded"]), class_weights)}
        print("Pesos calculados para las clases:")
        print(class_weight_dict)

        # Aplicar One-Hot solo al de entrenamiento
        self.labels_one_hot_train = keras.utils.to_categorical(self.train_df["label_encoded"], num_classes=self.NUM_CLASSES)
        self.labels_one_hot_test = keras.utils.to_categorical(self.test_df["label_encoded"], num_classes=self.NUM_CLASSES)

    # Implementación de Focal Loss para manejar mejor el desbalance
    def focal_loss(self,gamma=2.0, alpha=0.25):
        def focal_loss_fn(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Calcular focal loss
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = tf.pow(1 - y_pred, gamma) * y_true
            focal = alpha * weight * cross_entropy
            
            return tf.reduce_sum(focal, axis=-1)
        return focal_loss_fn

    # Crear modelo CNN mejorado
    def create_cnn_model(self, input_shape=(IMG_SIZE, IMG_SIZE, CHANNEL_SIZE), num_classes=None):
        # Determine number of classes from the training data if not provided
        if num_classes is None:
            num_classes = len(self.train_df["benign_malignant"].cat.categories)
        
        model = models.Sequential([
            # Normalización de entrada (importante)
            layers.Rescaling(1./255, input_shape=input_shape),
            
            # Primera capa de convolución
            layers.Conv2D(CONV_FILTER["conv1"], (3, 3), activation='relu', padding='same', 
                        kernel_regularizer=regularizers.l2(L2_LAMBDA)),
            layers.BatchNormalization(),
            layers.Conv2D(CONV_FILTER["conv1"], (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Segunda capa de convolución
            layers.Conv2D(CONV_FILTER["conv2"], (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(L2_LAMBDA)),
            layers.BatchNormalization(),
            layers.Conv2D(CONV_FILTER["conv2"], (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Tercera capa de convolución
            layers.Conv2D(CONV_FILTER["conv3"], (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(L2_LAMBDA)),
            layers.BatchNormalization(),
            layers.Conv2D(CONV_FILTER["conv3"], (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Cuarta capa de convolución
            layers.Conv2D(CONV_FILTER["conv4"], (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(L2_LAMBDA)),
            layers.BatchNormalization(),
            layers.Conv2D(CONV_FILTER["conv4"], (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Quinta capa de convolución (nueva)
            layers.Conv2D(CONV_FILTER["conv5"], (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(L2_LAMBDA)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Aplanar las características
            layers.Flatten(),
            
            # Capas densas (fully connected) con regularización
            layers.Dense(NEURONS_DENSE, activation='relu', 
                    kernel_regularizer=regularizers.l2(L2_LAMBDA)),
            layers.BatchNormalization(),
            layers.Dropout(DROPOUT_RATE),
            
            layers.Dense(NEURONS_DENSE2, activation='relu',
                    kernel_regularizer=regularizers.l2(L2_LAMBDA)),
            layers.BatchNormalization(),
            layers.Dropout(DROPOUT_RATE2),
            
            # Capa de salida
            layers.Dense(num_classes, activation='softmax')
        ])
        print("Modelo mejorado creado correctamente")
        return model

    # Función para cargar un bloque de imágenes con preprocesamiento mejorado
    def load_image_block(self, df, start_idx, end_idx, folder, labels_one_hot):
        images = []
        labels = []
        
        # Tomar solo las filas del rango especificado
        block_df = df.iloc[start_idx:end_idx]
        
        for index, row in block_df.iterrows():
            img_path = os.path.join(folder, row["isic_id"] + ".jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Preprocesamiento mejorado
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    # Normalización mejorada en el modelo, no aquí
                    
                    # Opcional: Mejorar contraste
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                    
                    images.append(img)
                    labels.append(labels_one_hot[index])
        
        return np.array(images), np.array(labels)

    # Función para cargar todas las imágenes de prueba con preprocesamiento mejorado
    def load_test_images(self, folder, df, labels_one_hot):
        images = []
        labels = []
        for index, row in df.iterrows():
            img_path = os.path.join(folder, row["isic_id"] + ".jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Preprocesamiento mejorado
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    # Normalización en el modelo
                    
                    # Opcional: Mejorar contraste
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                    
                    images.append(img)
                    labels.append(labels_one_hot[index])
        return np.array(images), np.array(labels)

    def validate_model(self):
        # Crear conjuntos de validación desde los datos de prueba
        self.X_test, self.y_test = self.load_test_images(TEST_DIR, self.test_df, self.labels_one_hot_test)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=42, 
                                                    stratify=np.argmax(self.y_test, axis=1))

        # Obtener el número total de imágenes en el conjunto de entrenamiento
        self.total_images = len(self.train_df)
        self.num_blocks = math.ceil(self.total_images / BLOCK_SIZE)

        print(f"Total de imágenes de entrenamiento: {self.total_images}")
        print(f"Entrenando en {self.num_blocks} bloques de {BLOCK_SIZE} imágenes cada uno")

        # Aplicar aumentación de datos mejorada específica para dermoscopia
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.2),
            layers.GaussianNoise(0.03),  # Ruido sutil para mejorar robustez
        ])

        # Callbacks mejorados para monitoreaar AUC además de accuracy
        self.callbacks = [
            EarlyStopping(monitor='val_auc', patience=8, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=4, min_lr=1e-6, mode='max'),
            ModelCheckpoint('isic_cnn_model_best.h5', save_best_only=True, monitor='val_auc', mode='max')
        ]

    def compile_model(self):
        # Crear y compilar el modelo con métricas más completas
        model = self.create_cnn_model(num_classes = self.NUM_CLASSES)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),  # Learning rate reducido
            # Usar focal loss 
            loss = self.focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                'accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        # Resumen del modelo
        model.summary()

        # Crear dataset de validación
        val_ds = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Función para aplicar aumentación de datos en cada lote
        def apply_augmentation(x, y):
            return self.data_augmentation(x, training=True), y

        # Entrenar el modelo por bloques
        history_per_block = []
        for block in range(self.num_blocks):
            start_idx = block * BLOCK_SIZE
            end_idx = min((block + 1) * BLOCK_SIZE, self.total_images)
            
            print(f"\n--- Entrenando bloque {block+1}/{self.num_blocks} (imágenes {start_idx} a {end_idx-1}) ---")
            
            # Cargar bloque de imágenes
            X_train_block, y_train_block = self.load_image_block(
                self.train_df, start_idx, end_idx, TRAIN_DIR, self.labels_one_hot_train
            )
            
            if len(X_train_block) == 0:
                print(f"No se encontraron imágenes en el bloque {block+1}. Saltando...")
                continue
            
            print(f"Imágenes cargadas en este bloque: {X_train_block.shape}")
            
            # Calcular pesos de clases para este bloque
            y_indices_block = np.argmax(y_train_block, axis=1)
            unique_classes = np.unique(y_indices_block)
            
            # Solo calcular pesos si hay más de una clase en el bloque
            if len(unique_classes) > 1:
                class_weights_block = compute_class_weight('balanced', classes=unique_classes, y=y_indices_block)
                class_weight_dict_block = {i: weight for i, weight in zip(unique_classes, class_weights_block)}
                print("Pesos de clase para este bloque:", class_weight_dict_block)
            else:
                # Si solo hay una clase en el bloque, usar pesos predeterminados
                class_weight_dict_block = class_weight_dict
                print("Usando pesos de clase predeterminados para este bloque")
            
            # Crear dataset de TensorFlow para este bloque con aumentación de datos
            train_ds_block = tf.data.Dataset.from_tensor_slices((X_train_block, y_train_block))
            train_ds_block = train_ds_block.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
            train_ds_block = train_ds_block.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            
            # Entrenar el modelo con este bloque
            print(f"Entrenando bloque {block+1}...")
            history_block = model.fit(
                train_ds_block,
                epochs=EPOCHS,
                validation_data=val_ds,
                callbacks = self.callbacks,
                class_weight=class_weight_dict_block,
                verbose=1
            )
            
            history_per_block.append(history_block)
            print(f"Bloque {block+1} completado.")

    def save_model(self):
        version = self.settings.get_actual_version()
        self.model_name = f"/isic_cnn_model_v{version}.h5"
        
        # Guardar modelo entrenado
        self.model.save(MODEL_DIR + self.model_name)
        print("Modelo guardado correctamente")

    # --------------------------------------------------------

    
        
    def save_settings(self):
        new_input = ({
            "archivo": self.model_name,
            "fecha_entrenamiento": "2025-16-01",
            "imagenes": self.total_images,
            "epocas": EPOCHS,
            "batch_size": BATCH_SIZE,
            "neurons": NEURONS_DENSE,
            "neurons2": NEURONS_DENSE2,
            "dropout": DROPOUT_RATE,
            "dropout2": DROPOUT_RATE2,
            "convolutional_layers": {
                "conv1": CONV_FILTER["conv1"],
                "conv2": CONV_FILTER["conv2"],
                "conv3": CONV_FILTER["conv3"],
                "conv4": CONV_FILTER["conv4"],
                "conv5": CONV_FILTER["conv5"]
            },
            
            "accuracy": self.accuracy,
            "loss": self.loss,
            "f1": self.f1,
            "recall": self.recall,
            "matrix_confusion": self.confusion_matrix
        })
        
        self.settings.new_version(new_input)