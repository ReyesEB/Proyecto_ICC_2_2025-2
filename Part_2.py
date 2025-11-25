import cv2

img_array = cv2.imread("dataset/0.jpeg",cv2.IMREAD_GRAYSCALE)

print(img_array)

nueva_imagen = cv2.resize(img_array,(8,8))

print(nueva_imagen,"\n")

print("invertimos la escala:\n")

for i in range(8):
    for j in range(8):
        nueva_imagen[i,j] = 255-nueva_imagen[i,j]
print(nueva_imagen,"\n")

print("Achatamiento:\n")

for i in range(8):
    for j in range(8):
        nueva_imagen[i,j] = (nueva_imagen[i,j]/255)*16
print(nueva_imagen)

# EJEMPLO GPT
from sklearn.neighbors import KNeighborsClassifier

# Datos de ejemplo (características)
X = [
    [1, 2],
    [2, 3],
    [3, 1],
    [6, 7],
    [7, 8],
    [8, 6]
]

# Etiquetas de esos datos
y = ['A', 'A', 'A', 'B', 'B', 'B']

# Creamos el modelo con K=3
modelo = KNeighborsClassifier(n_neighbors=3)

# Entrenamos
modelo.fit(X, y)

# Predecimos para un punto nuevo
print(modelo.predict([[4, 3]]))