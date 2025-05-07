import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Paso 1: Generar datos de ejemplo (50 niños con calificaciones entre 0 y 5 en 4 habilidades)
np.random.seed(42) # Para reproducibilidad
ninos = 50 # Número de niños
habilidades = 4 # Número de habilidades

# Generamos calificaciones aleatorias (valores entre 0 y 5)
calificaciones = np.random.uniform(0, 5, (ninos, habilidades))

# Paso 2: Aplicar Fuzzy C-Means con 4 clústeres
n_clust = 4 # Ahora usamos 4 clústeres

# Ejecutamos el algoritmo FCM
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    calificaciones.T, n_clust, 2, error=0.005, maxiter=1000, init=None
)

# Paso 3: Reducción a 2 dimensiones para visualización
pca = PCA(n_components=2)
calificaciones_2d = pca.fit_transform(calificaciones)

# Asignar a cada niño el clúster con mayor grado de pertenencia
clustering = np.argmax(u, axis=0)

# Paso 4: Graficar los clústeres
plt.figure(figsize=(8, 6))

# Colores diferentes para cada clúster
colores = ['r', 'g', 'b', 'c']

for i in range(n_clust):
    plt.scatter(calificaciones_2d[clustering == i, 0],
    calificaciones_2d[clustering == i, 1],
    color=colores[i], label=f'Clúster {i+1}', alpha=0.6)

plt.title("Clustering Fuzzy C-Means de habilidades matemáticas de niños")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.show()