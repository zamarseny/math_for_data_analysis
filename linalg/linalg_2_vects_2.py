import sys


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    n = int(input())
    u = [float(i) for i in input().split()]
    v = [float(i) for i in input().split()]
    dot_prod = 0.0
    for i in range(n):
        dot_prod += u[i] * v[i]
    if dot_prod == 0.0:
        print("ORTHOGONAL")
    else:
        print("NON-ORTHOGONAL")


if __name__ == "__main__":
    main()
