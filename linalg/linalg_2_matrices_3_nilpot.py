import sys


def read_matrix():
    n = int(input())
    matr_a = []
    for i in range(n):
        row_a = [float(val) for val in input().split()]
        matr_a.append(row_a)
    return matr_a, n


def matr_multiplication(a: list, b: list) -> list:
    m = len(a)
    h = len(b)
    k = len(b[0])
    c = []
    for i in range(m):
        c_i = []
        for j in range(k):
            c_i_j = 0.0
            for l in range(h):
                c_i_j += a[i][l] * b[l][j]
            c_i.append(c_i_j)
        c.append(c_i)
    return c


def is_null_m(matr, n):
    for i in range(n):
        for j in range(n):
            if matr[i][j] != 0:
                return False
    return True


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    nilpotent_power = 100
    matr_a, n = read_matrix()
    matr_product = matr_a.copy()
    k = 1
    while k <= n:
        if is_null_m(matr_product, n):
            print(k)
            return
        matr_product = matr_multiplication(matr_product, matr_a)
        k += 1
    print(nilpotent_power)


if __name__ == "__main__":
    main()
