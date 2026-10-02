import numpy as np

m1 = np.arange(1, 7).reshape(2, 3)
print(m1)

m2 = np.arange(7, 13).reshape(3, 2)
print(m2)

print(m1 @ m2)

print(m2 @ m1)
