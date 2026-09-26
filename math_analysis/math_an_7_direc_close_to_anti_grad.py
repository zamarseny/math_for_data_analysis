import math
import sys


def load_func_coefficients(f_in):
    n, k, eps = [float(i) for i in f_in.readline().split()]
    n = int(n)
    k = int(k)
    grad = [float(i) for i in f_in.readline().split()]
    directions = []
    for i in range(k):
        direc = [float(i) for i in f_in.readline().split()]
        directions.append(direc)
    return n, k, eps, grad, directions


def scalar_product(a, b):
    sc_product = 0
    for i in range(len(a)):
        sc_product += a[i] * b[i]
    return sc_product


def vector_lenght(a):
    lenght = 0
    for i in range(len(a)):
        lenght += a[i] ** 2
    lenght = math.sqrt(lenght)
    return lenght


def closest_to_antigrad(directions, grad, eps):
    best_idx = 0
    nabla_l_min = float("inf")
    for i in range(len(directions)):
        length = vector_lenght(directions[i])
        if length == 0:
            continue  # division by zero
        nabla_l = scalar_product(directions[i], grad) * eps / length
        if nabla_l < nabla_l_min:
            nabla_l_min = nabla_l
            best_idx = i + 1
    return best_idx, nabla_l_min


def main():
    n, k, eps, grad, directions = load_func_coefficients(sys.stdin)

    best_idx, nabla_l = closest_to_antigrad(directions, grad, eps)

    print(f"{best_idx} {nabla_l:.4f}")


if __name__ == "__main__":
    main()
