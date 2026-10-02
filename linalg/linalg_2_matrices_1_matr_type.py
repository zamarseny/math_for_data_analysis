import sys


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    upper_triang = True
    lower_triang = True
    n = int(input())
    for i in range(n):
        string_row = input().split()
        for num, val in enumerate(string_row):
            if num < i and float(val) != 0:
                upper_triang = False
            if num > i and float(val) != 0:
                lower_triang = False
    if lower_triang and upper_triang:
        print("DIAGONAL")
    elif lower_triang:
        print("LOWER_TRIANGULAR")
    elif upper_triang:
        print("UPPER_TRIANGULAR")
    else:
        print("OTHER")


if __name__ == "__main__":
    main()
