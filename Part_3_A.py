import pandas as pd

Datos = pd.read_csv("resultados.csv")

valores_reales = Datos["Valor real"]
valores_predichos = Datos["Clasificacion de la IA"]


for i in range(10):
    tp = tn = fp = fn = 0

    for j in range(len(valores_reales)):
        real = valores_reales[j]
        pred = valores_predichos[j]
        if real == i and pred == i:
            tp += 1
        elif real == i and pred != i:
            fn += 1
        elif real != i and pred == i:
            fp += 1
        elif real != i and pred != i:
            tn += 1
    matriz = pd.DataFrame([[tp,fn],[fp,tn]],columns=[f"Es {i}",f"No es {i} (prediccion)"],index=[f"Es {i}",f"No es {i} (real)"])
    print(matriz)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    print("Precision: ", precision,"|","Recall: ", recall,"|","Accuracy: ", accuracy,"|","F1-score: ", f1_score)
    print("-------------------------------------------------------------------------------------------")
