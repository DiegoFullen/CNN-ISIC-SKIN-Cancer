import pandas as pd

# Cargar datos
df = pd.read_csv("metadata-training.csv")  # Reemplaza con tu archivo

# Revisar valores únicos en la columna de clasificación
print(df["benign_malignant"].value_counts())
