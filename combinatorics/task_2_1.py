import sys


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    input: 2 10 26, expected: 520
    input: 10 2 3, expected: 9 705 552 (got 210 937 500)
    """
    l, a, b = [int(i) for i in input().split()]
    rest = l - 2 if l > 1 else 0

    print(rest)

    # n = a * l * b * (l - 1) * (a + b) ** rest
    n = (a + b) ** l - a**l - b**l
    print(n)

    def generate(l, a, b):
        a_figures = set([i for i in range(a)])
        b_letters = set([j for j in range(a, b + a)])
        for i in range(l):
            comb = 1


if __name__ == "__main__":
    main()
