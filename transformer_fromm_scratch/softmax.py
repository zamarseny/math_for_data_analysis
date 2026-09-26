import math


def softmax_naive(x):
    result = []
    denom = 0.0
    for i in range(len(x)):
        nom = math.exp(x[i])
        denom += math.exp(x[i])
        result.append(nom)
    result = [r / denom for r in result]
    return result


def softmax_vectorized(x):
    result = [math.exp(x[i]) for i in range(len(x))]
    denom = sum(result)
    result = [r / denom for r in result]
    return result


print(softmax_naive([1, 2, 3]))
print(softmax_vectorized([1, 2, 3]))
print(softmax_naive([1.3, 5.1, 2.2, 0.7, 1.1]))
print(softmax_vectorized([1.3, 5.1, 2.2, 0.7, 1.1]))
