import sys


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    with open("input.txt", "r") as file:
        # m = int(input())
        m = int(file.readline())
        # counts = (int(i) for i in input().split())
        counts = [int(i) for i in file.readline().split()]

    def factor(x):
        prod = 1
        if x > 0:
            for i in range(1, x + 1):
                prod *= i
        return prod

    numer_sum = sum(i for i in counts)
    numer = factor(numer_sum)
    denomin = 1
    for cou in counts:
        denomin *= factor(cou)
    # print(numer)
    # print(denomin, [i for i in counts])
    n = numer / denomin
    print(n)


if __name__ == "__main__":
    main()
