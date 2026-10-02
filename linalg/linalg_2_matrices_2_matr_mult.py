import sys


def read_matrices():
    m, n = [int(i) for i in input().split()]
    matr_a = []
    for i in range(m):
        row_a = [float(i) for i in input().split()]
        matr_a.append(row_a)

    h, k = [int(i) for i in input().split()]
    matr_b = []
    for i in range(h):
        row_b = [float(i) for i in input().split()]
        matr_b.append(row_b)
    return matr_a, matr_b, n, h


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


def print_res(c):
    for row in c:
        print(f"{' '.join([str(int(i)) for i in row])}")


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    matr_a, matr_b, n, h = read_matrices()
    if n != h:
        print("NOT_DEFINED")
    else:
        matr_c = matr_multiplication(matr_a, matr_b)
        print_res(matr_c)


if __name__ == "__main__":
    main()
