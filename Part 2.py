import cv2

img_array = cv2.imread("Datasets/miNumero.png",cv2.IMREAD_GRAYSCALE)

print(img_array)

nueva_imagen = cv2.resize(img_array,(8,8))

print(nueva_imagen,"\n")

print("invertimos la escala:\n")

for i in range(8):
    for j in range(8):
        nueva_imagen[i,j] = 255-nueva_imagen[i,j]
print(nueva_imagen,"\n")

print("Achatamiento:\n")

for i in range(8):
    for j in range(8):
        nueva_imagen[i,j] = (nueva_imagen[i,j]/255)*16
print(nueva_imagen)