import pandas as pd
import numpy as np

Datos = pd.read_csv("resultados.csv")

valores_reales = Datos["Valor real"]
valores_predichos = Datos["Clasificacion de la IA"]
motivo = Datos["Motivo de la clasificacion"]


for i in range(10):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j in range(len(valores_reales)):
        if valores_reales[j] == i:
            if valores_predichos[j] == valores_reales[j] and motivo[j] == 1:
                tp += 1
            elif valores_predichos[j] == valores_reales[j] and motivo[j] == 0:
                fn += 1
            elif valores_predichos[j] != valores_reales[j] and motivo[j] == 1:
                fp += 1
            else:
                tn += 1
    matriz = pd.DataFrame([[tp,fn],[fp,tn]],columns=[f"Es {i}",f"No es {i} (prediccion)"],index=[f"Es {i}",f"No es {i} (real)"])
    print(matriz)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    print("Precision: ", precision,"|","Recall: ", recall,"|","Accuracy: ", accuracy,"|","F1-score: ", f1_score)
    print("-------------------------------------------------------------------------------------------")
