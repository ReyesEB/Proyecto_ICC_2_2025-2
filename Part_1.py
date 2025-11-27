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
los_3_cercanos = []
resultados_ia = []
motivo_de_la_clasificacion = []
archivo_leido = []
for archivo in os.listdir(ruta): #archivo tiene la forma de: "num"-img"num".png
    trans_cadena = str(archivo)
    archivo_leido.append(archivo)
    split = trans_cadena.split("-")
    valor_real.append(int(split[0]))
    ruta_completa = os.path.join(ruta, archivo)
    img_array = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)

    nueva_imagen = cv2.resize(img_array,(8,8))

    nueva_imagen = 255 - nueva_imagen

    for i in range(8):
        for j in range(8):
            nueva_imagen[i,j] = (nueva_imagen[i,j]/255)*16
    print(ruta_completa)
    print(int(split[0]))
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
    lista_euclidiana = distancia_euclidiana(nueva_imagen)


    lista_temp = []
    for _ in range(3):
        minimo = lista_euclidiana.index(min(lista_euclidiana))
        lista_temp.append(int(target[minimo]))
        print(f"La distancia {_+1} es {lista_euclidiana[minimo]}")
        print(target[minimo])
        lista_euclidiana.remove(lista_euclidiana[minimo])
        print(minimo)
    los_3_cercanos.append(lista_temp)

    repetido = None

    for elemento in lista_temp:
        if lista_temp.count(elemento) > 1:
            repetido = elemento  # solo se guarda, no se corta el ciclo

    if repetido is not None:
        print("Soy la inteligencia artificial, y he detectado que el digito ingresado corresponde al numero:", repetido)
        resultados_ia.append(repetido)
        motivo_de_la_clasificacion.append(1)# 1 porque lo detecto de porque se repetian minimo 2
    else:
        print("Soy la inteligencia artificial, como no detecto 2 o mas similares, regreso el elemento mas cercano:", lista_temp[0])
        resultados_ia.append(lista_temp[0])
        motivo_de_la_clasificacion.append(0)# 0 porque ninguno se repetia mas de 2 veces

resultado = np.column_stack((archivo_leido,valor_real, los_3_cercanos, resultados_ia, motivo_de_la_clasificacion))
Dataframe = pd.DataFrame(resultado, columns=["Archivo","Valor real","Primer cercano","Segundo cercano","Tercer cercano","Clasificacion de la IA","Motivo de la clasificacion"])
Dataframe.to_csv("resultados.csv",index=False)