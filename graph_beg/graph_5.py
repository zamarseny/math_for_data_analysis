import sys


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    with open("input.txt", "r") as f_in:
        n = int(f_in.readline())
    with open("output.txt", "w") as f_out:
        f_out.write(str(int(n * (n - 1) / 2)))


if __name__ == "__main__":
    main()
