import numpy as np
import pandas as pd
from sklearn import datasets
import cv2
import os
# aves sin nido

dataset=datasets.load_digits()

target = dataset["target"]
images = dataset["images"]

ruta = "los_numeros"
valor_real = []

for archivo in os.listdir(ruta): #archivo tiene la forma de: "num"-img"num".png
    trans_cadena = str(archivo)
    split = trans_cadena.split("-")
    valor_real.append(int(split[0]))
    ruta_completa = os.path.join(ruta, archivo)
    img_array = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)

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
print(valor_real)

