import pandas as pd

Datos = pd.read_csv("resultados.csv")

valores_reales = Datos["Valor real"]
valores_predichos = Datos["Clasificacion de la IA"]
