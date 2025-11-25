import pandas as pd
from sklearn import datasets

# aves sin nido

dataset=datasets.load_digits()
print(dataset)
target = dataset["target"]
images = dataset["images"]
print(images)

def distancia_euclidiana(a):
    lista = []
    for i in range(1797):
        suma = 0
        for j in range(8):
            for k in range(8):
                suma += (images[i][j][k] - a[j][k])**2
        distancia = suma ** 0.5
        lista.append(distancia)
    return lista

dataframe = pd.DataFrame(target, columns=["Etiqueta/target"])
print(dataframe)

valores_reales = []