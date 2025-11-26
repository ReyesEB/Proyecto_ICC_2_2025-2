import numpy as np
import pandas as pd
from sklearn import datasets
import cv2
# aves sin nido

dataset=datasets.load_digits()
print(dataset)
target = dataset["target"]
images = dataset["images"]
print(images)

img_array = cv2.imread("dataset/9-img49.png",cv2.IMREAD_GRAYSCALE)

nueva_imagen = cv2.resize(img_array,(8,8))

nueva_imagen = 255 - nueva_imagen

for i in range(8):
    for j in range(8):
        nueva_imagen[i,j] = (nueva_imagen[i,j]/255)*16
print(nueva_imagen)

def distancia_euclidiana(a):
    lista = []
    for i in range(1797):
        suma=0
        suma += np.sum((images[i]-a)**2)
        distancia = suma ** 0.5
        distancia = float(distancia)
        lista.append(distancia)
    return lista

print(distancia_euclidiana(nueva_imagen))

dataframe = pd.DataFrame(target, columns=["Etiqueta/target"])
print(dataframe)
