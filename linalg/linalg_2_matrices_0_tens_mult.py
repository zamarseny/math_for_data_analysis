import numpy as np

# Создаём два 3D массива (тензора)
X = np.random.rand(2, 3, 4)
Y = np.random.rand(2, 4, 5)

# np.matmul (или оператор @) умножает X и Y по последним двум измерениям
matmul_result = X @ Y
print("Форма результата матричного умножения:", matmul_result.shape)

# Поэлементное умножение (*) здесь не сработает напрямую,
# т. к. размеры (2,3,4) и (2,4,5) несовместимы для
# поэлементного умножения без явного преобразования.
elementwise_result = X * Y

# Output:
# Форма результата матричного умножения: (2, 3, 5)
# ValueError: operands could not be broadcast together with shapes (2,3,4) (2,4,5)
