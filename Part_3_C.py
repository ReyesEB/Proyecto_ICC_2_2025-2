import pandas as pd

Datos = pd.read_csv("resultados.csv")

valores_reales = Datos["Valor real"]
valores_predichos = Datos["Clasificacion de la IA"]

for i in range(10):  # Para cada dígito
    # Filtramos SOLO los casos donde el valor real es i
    indices = valores_reales[valores_reales == i].index

    if len(indices) == 0:
        continue  # Si no hay ejemplos de ese dígito, lo saltamos

    tp = fn = 0

    # Revisamos solo dentro de este grupo
    for j in indices:
        if valores_predichos[j] == i:
            tp += 1
        else:
            fn += 1

    # Creamos matriz 2x2 con solo TP y FN
    matriz = pd.DataFrame(
        [[tp, fn]],
        columns=[f"Predice {i}", f"Predice ≠ {i}"],
        index=[f"Real {i}"]
    )

    # Métricas (solo precisión en este ítem)
    precision = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("=================================================")
    print(f"Matriz de confusión (dígito {i})")
    print(matriz)
    print(f"Precisión para el dígito {i}: {precision}")
