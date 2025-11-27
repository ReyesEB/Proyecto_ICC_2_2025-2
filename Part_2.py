import pandas as pd
from sklearn.metrics import confusion_matrix

# Importamos lo que queremos:
dtf = pd.read_csv("resultados.csv")
valores_reales = dtf['Valor real']
valor_predicho = dtf['Clasificacion de la IA']

# Usamos el confusion_matrix
matriz = confusion_matrix(valores_reales, valor_predicho)
print(matriz)

numeros_predichos = [i for i in range(10)]
Dataframe = pd.DataFrame(matriz,columns=numeros_predichos,index=numeros_predichos)
print(Dataframe)

print()
Dataframe.to_csv("matriz_confusion_10_x_10.csv")
