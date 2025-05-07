import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from Utils.evaluation import evaluate # Clase custom

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
keras = tf.keras

# ------------------------ HYPERPARAMETERS / CONFIGURACIÓN ----------------------------

# Rutas
TRAIN_DIR = "Training2"
TEST_DIR = "Test2"
TEST_CSV_FILE = "datasets/metadata-test copy.csv"  # Ruta del archivo con etiquetas
TRAIN_CSV_FILE = "datasets/metadata-training copy.csv"  # Ruta del archivo con etiquetas

#Hyperparámetros

IMG_SIZE = 224
CHANNEL_SIZE = 3  # Número de canales de la imagen (RGB)

# CNN Hyperparameters
EPOCHS = 20
BATCH_SIZE = 32
NEURONS_DENSE = 512  # Número de neuronas en la capa densa
NEURONS_DENSE2 = 256  # Número de neuronas en la segunda capa densa
DROPOUT_RATE = 0.5  # Tasa de dropout para prevenir overfitting (Desactiva un porcentaje de neuronas)
DROPOUT_RATE2 = 0.3  
CONV_FILTER = {
    "conv1": 32,
    "conv2": 64,
    "conv3": 128,
    "conv4": 256,
    }

# Pesos para clases: 0 = benign, 1 = malignant
CLASS_WEIGHT_BENIGN = 1.0
CLASS_WEIGHT_MALIGNANT = 2.0
class_weight_dict = {0: CLASS_WEIGHT_BENIGN, 1: CLASS_WEIGHT_MALIGNANT}

# Separacion de datos
SEPARATION = 1000  # Imagemes por bloque de entreno
# ------------------------------------------------------------------------------------

# Cargar el CSV para entrenamiento y prueba
train_df = pd.read_csv(TRAIN_CSV_FILE) 
test_df = pd.read_csv(TEST_CSV_FILE)    

# Preprocesar ambos dataframes
for df in [train_df, test_df]:
    df["benign_malignant"] = df["benign_malignant"].astype("category")
    df["label_encoded"] = df["benign_malignant"].cat.codes

# Aplicar One-Hot solo al de entrenamiento
labels_one_hot_train = keras.utils.to_categorical(train_df["label_encoded"])
labels_one_hot_test = keras.utils.to_categorical(test_df["label_encoded"])
    
# Función para cargar imágenes
def load_images_from_folder(folder, df, labels_one_hot):
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = os.path.join(folder, row["isic_id"] + ".jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img)
            labels.append(labels_one_hot[index])
    return np.array(images), np.array(labels)

# Cargar imágenes
X_train, y_train = load_images_from_folder(TRAIN_DIR, train_df, labels_one_hot_train)
X_test, y_test = load_images_from_folder(TEST_DIR, test_df, labels_one_hot_test)

print(f"Imágenes de entrenamiento: {X_train.shape}, Etiquetas: {y_train.shape}")
print(f"Imágenes de prueba: {X_test.shape}, Etiquetas: {y_test.shape}")


NUM_CLASSES = len(train_df["label_encoded"].unique()) # Número de clases

# Calcular pesos de clases para manejar el desbalance
# Primero convertir y_train de one-hot a índices de clase
y_indices = np.argmax(y_train, axis=1)
unique_classes = np.unique(y_indices)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_indices)
class_weight_dict = {i: weight for i, weight in zip(unique_classes, class_weights)}

print("Pesos de clase:", class_weight_dict)

# Crear modelo CNN
def create_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, CHANNEL_SIZE), num_classes=NUM_CLASSES):
    
    """
        32, 64, 128, 256: Número de filtros en cada capa.
        (3, 3): Tamaño del filtro.
        activation='relu': Función de activación ReLU (Rectified Linear Unit), que introduce no linealidad.
        padding='same': Mantiene el tamaño de la imagen de entrada al agregar ceros alrededor de la imagen.
    """
    model = models.Sequential([
        # Primera capa de convolución
        layers.Conv2D(CONV_FILTER["conv1"], (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(), # Normaliza las salidas de la capa anterior
        layers.MaxPooling2D((2, 2)), # Reduce la dimensionalidad de las características (2x2)
        
        # Segunda capa de convolución
        layers.Conv2D(CONV_FILTER["conv1"], (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(), 
        layers.MaxPooling2D((2, 2)),
        
        # Tercera capa de convolución
        layers.Conv2D(CONV_FILTER["conv1"], (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Cuarta capa de convolución
        layers.Conv2D(CONV_FILTER["conv1"], (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Aplanar las características
        layers.Flatten(),
        
        # Capas densas (fully connected)
        layers.Dense(NEURONS_DENSE, activation='relu'),
        layers.Dropout(DROPOUT_RATE), 
        layers.Dense(NEURONS_DENSE2, activation='relu'),
        layers.Dropout(DROPOUT_RATE2),
        
        # Capa de salida
        layers.Dense(num_classes, activation='softmax')
    ])
    print("modelo terminado")
    return model

# Crear y compilar el modelo
model = create_cnn_model()
print("Compilando modelo")
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Resumen del modelo
print("Resumen del modelo: ")
model.summary()

# Callbacks para mejorar el entrenamiento
print("Callbacks del entrenamiento")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint('isic_cnn_model.h5', save_best_only=True, monitor='val_accuracy')
]

# Aumentación de datos para mejorar la generalización
print("Aumentando la generalizacion")
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# Crear conjuntos de validación
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

# Aplicar aumentación de datos en cada lote
def apply_augmentation(x, y):
    return data_augmentation(x, training=True), y

# Crear datasets de TensorFlow
print("Creando datasets de tensorflow")
train_ds = tf.data.Dataset.from_tensor_slices((X_train_split, y_train_split))
train_ds = train_ds.map(apply_augmentation).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Entrenar el modelo
print("Entrenando al modelo")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Guardar modelo entrenado
model.save('Model/isic_cnn_model.keras')

# --------------------------------------------------------

# ------------- Evaluaciones del modelo ------------------

# Hacer evaluación del modelo
evaluate = evaluate(model, test_ds, train_df["benign_malignant"].cat.categories.tolist())
evaluate.evaluate_model()
evaluate.plot_training_history(history)

# Hacer predicciones en datos de prueba
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(np.concatenate([y for x, y in test_ds], axis=0), axis=1)

# Obtener los nombres de las clases
class_names = train_df["benign_malignant"].cat.categories.tolist() 

# Mostrar la matriz de confusión
evaluate.matrix_confusion(true_classes, predicted_classes)

# Visualizar algunas predicciones
num_images = 5 # Tomar 5 imágenes de prueba
sample_images = X_test[:num_images]  
sample_predictions = model.predict(sample_images)

# Mostrar el reporte de clasificación
evaluate.display_predictions(sample_images, sample_predictions, y_test[:num_images])
