import numpy as np
x = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(x)
y = np.array([[3,6,5,4],[5,6,7,8],[9,10,11,12]])
print((x-y)**2)
z = np.sum((x-y)**2)
print(z**0.5)