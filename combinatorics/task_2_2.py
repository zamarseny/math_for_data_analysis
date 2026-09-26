import sys
import itertools

def brute(n, k):
    vals = [i for i in range(n)]
    sets = []
    for i in range(k):
        for j in range(n):
    return len(sets)

def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """

    def factor(n):
        fact = 1
        if n > 1:
            for i in range(1, n + 1):
                fact *= i
        return fact

    def c_n_k(n, k):
        return factor(n) / factor(k) / factor(n - k)

    n, k = [int(i) for i in input().split()]
    if n == 1:
        print("1")
    elif n * k > 0:
        combs_init = n**k
        combs = combs_init
        for i in range(1, k):
            print(f"interm: {combs}")
            combs -= c_n_k(n, i)
        print(int(combs))
    else:
        print("0")


if __name__ == "__main__":
    main()
