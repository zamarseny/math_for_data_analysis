import sys


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    a, b, c, ab, ac, bc, u = [int(i) for i in input().split()]
    a_only, b_only, c_only, ab_only, ac_only, bc_only, abc = [i for i in range(11, 18)]
    abc = u - a - b - c + ab + ac + bc
    ab_only = ab - abc
    ac_only = ac - abc
    bc_only = bc - abc
    a_only = a - ab - ac + abc
    b_only = b - ab - bc + abc
    c_only = c - ac - bc + abc
    if (
        a_only < 0
        or b_only < 0
        or c_only < 0
        or ab_only < 0
        or ac_only < 0
        or bc_only < 0
        or abc < 0
    ):
        print("INCONSISTENT")
    else:
        print(a_only, b_only, c_only, ab_only, ac_only, bc_only, abc)


if __name__ == "__main__":
    main()
