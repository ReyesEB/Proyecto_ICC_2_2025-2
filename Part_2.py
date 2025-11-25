

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