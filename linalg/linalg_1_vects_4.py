import sys
import math


def scalar_prod(v1, v2):
    dot_prod = sum([v1[i] * v2[i] for i in range(len(v1))])
    return dot_prod


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    """
    n = int(input())
    v_1 = [int(i) for i in input().split()]
    v_2 = [int(i) for i in input().split()]
    angle = (
        math.acos(
            scalar_prod(v_1, v_2)
            / math.sqrt(scalar_prod(v_1, v_1) * scalar_prod(v_2, v_2))
        )
        * 180
        / math.pi
    )
    print(f"{(angle // 1):.0f}")


if __name__ == "__main__":
    main()
