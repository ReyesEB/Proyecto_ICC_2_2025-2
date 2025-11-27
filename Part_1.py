import numpy as np           # Importamos NumPy para manejo numérico
import pandas as pd          # Importamos Pandas para crear el CSV
from sklearn import datasets # Importamos datasets de sk-learn
import cv2                   # Importamos OpenCV para procesar imágenes
import os                    # Importamos OS para manejar rutas y archivos


dataset=datasets.load_digits() # Cargamos el dataset de dígitos (8x8) incluido en sklearn

target = dataset["target"]     # Etiquetas del dataset
images = dataset["images"]     # Imágenes del dataset

# Ponemos la ruta del archivo de las imagenes
ruta = "los_numeros"

# Almacenaremos los valores reales de los numeros
valor_real = []

# Aca estaran los 3 mas cercanos
los_3_cercanos = []

# Esto sera lo que la IA clasificara
resultados_ia = []

# Almacenara 0 y 1 dependiendo de como lo clasifico.
motivo_de_la_clasificacion = [] # 0 = solo el más cercano, 1 = al menos dos repetidos

# Es el nombre del archivo leido, lo usaremos para hacer el csv con cada resultado
archivo_leido = []


# Esta es la parte de las IMAGENES
for archivo in os.listdir(ruta):

    trans_cadena = str(archivo)   # Convertimos el nombre del archivo a cadena
    archivo_leido.append(archivo) # Guardamos el archivo en la lista

    # Cortamos la cadena con respecto al "-"
    split = trans_cadena.split("-") # Las fotos tienen la forma de: "num"-img"num".png

    # Vemos el valor real, el cual se encuentra antes del "-"
    valor_real.append(int(split[0]))

    #  En estya variable almacenamos la ruta completa del archivo
    ruta_completa = os.path.join(ruta, archivo)

    # Abrimos la imagen en escala de grises
    img_array = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)

    # Redimensionamos la imagen a 8x8 igual que las del dataset original
    nueva_imagen = cv2.resize(img_array,(8,8))

    print('Numero real:',int(split[0]))

    # Invertimos la escala
    nueva_imagen = 255 - nueva_imagen

    # Aplicamos el Martillaso, cambiando la escala de 0-255 a 0 - 16
    for i in range(8):
        for j in range(8):
            nueva_imagen[i,j] = (nueva_imagen[i,j]/255)*16

    print('Ubicacion de la imagen:',ruta_completa)
    print(nueva_imagen)

    # Funcion para el calculo de la distancia euclidiana
    def distancia_euclidiana(a):
        lista = []                              # Aquí guardaremos las distancias
        for i in range(1797):
            suma=0
            suma += np.sum((images[i]-a)**2)
            distancia = suma ** 0.5
            distancia = float(distancia)
            lista.append(distancia)             # Guardamos la distancia
        return lista

    # Calculamos todas las distancias con la imagen externa procesada
    lista_euclidiana = distancia_euclidiana(nueva_imagen)

    # Lista temporal donde guardaremos los indices de los 3 vecinos más cercanos
    lista_temp = []

    for _ in range(3):
        # Buscamos cual es el minimo indice encontrado en los 3 mas cercanos
        minimo = lista_euclidiana.index(min(lista_euclidiana))

        # Agregamos a la lista temporal el indice minimo detectado
        lista_temp.append(int(target[minimo]))
        print(f"La distancia {_+1} es {lista_euclidiana[minimo]}")
        print('Target detectado:',target[minimo])

        # Eliminamos esa distancia de la lista para no repetirla
        lista_euclidiana.remove(lista_euclidiana[minimo])

        print('El indice es:',minimo)

    # Guardamos los 3 más cercanos
    los_3_cercanos.append(lista_temp)

    repetido = None

    # Buscamos si hay algún elemento repetido al menos dos veces
    for elemento in lista_temp:
        if lista_temp.count(elemento) > 1:
            repetido = elemento  # solo se guarda, no se corta el ciclo

    # Si hay repetido ahi termina
    if repetido is not None:
        print("Soy la inteligencia artificial, y he detectado que el digito ingresado corresponde al numero:", repetido)

        # Y lo agregamos a los resultado de IA
        resultados_ia.append(repetido)

        motivo_de_la_clasificacion.append(1) # 1 porque lo detecto de porque se repetian minimo 2

    # Si no hubo repetidos, solo devolvemos el más cercano
    else:
        print("Soy la inteligencia artificial, como no detecto 2 o mas similares, regreso el elemento mas cercano:", lista_temp[0])

        # Lo mismo
        resultados_ia.append(lista_temp[0])

        motivo_de_la_clasificacion.append(0) # 0 porque ninguno se repetia mas de 2 veces
    print("------------------------------------------------------------------------------------------------------------------------")
    resultado = np.column_stack((archivo_leido,valor_real, los_3_cercanos, resultados_ia, motivo_de_la_clasificacion))
Dataframe = pd.DataFrame(resultado, columns=["Archivo","Valor real","Primer cercano","Segundo cercano","Tercer cercano","Clasificacion de la IA","Motivo de la clasificacion"])
Dataframe.to_csv("resultados.csv",index=False)