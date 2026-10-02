import numpy as np

r = 4  # скаляр
v = np.arange(3)  # вектор
a = np.arange(9).reshape(3, 3)  # матрица

# Сумма скаляра и вектора
print(r + v)

# Сумма вектора и матрицы
print(v + a)

# Сумма векторов разного размера
print(v.reshape(3, 1) + v)

# Output:
# array([4, 5, 6])

# array([[ 0,  2,  4],
#        [ 3,  5,  7],
#        [ 6,  8, 10]])

# array([[0, 1, 2],
#        [1, 2, 3],
#        [2, 3, 4]])
