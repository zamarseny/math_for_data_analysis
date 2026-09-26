import math


def calc_value(text_f, x):
    e = math.e
    return eval(text_f)


def is_continuous_lipshitz(text_f, x_0, sigma, eps) -> bool:
    x_right = x_0 + sigma
    value_right = calc_value(text_f, x_right)
    x_left = x_0 - sigma
    value_left = calc_value(text_f, x_left)
    value_center = calc_value(text_f, x_0)

    return (
        abs(value_right - value_center) < eps and abs(value_left - value_center) < eps
    )


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    with open("input.txt", "r") as f_in:
        text_f = f_in.readline()
        a, b = [float(i) for i in f_in.readline().split()]
        l = float(f_in.readline())

    eps = 1e-6

    print(text_f, a, b, l, calc_value(text_f, b))

    is_contin = True  # is_continuous_lipshitz(text_f, x_0, sigma, eps)

    # print(is_contin)

    with open("output.txt", "w") as f_out:
        if is_contin:
            f_out.write("LIPSCHITZ")
        else:
            f_out.write("NOT LIPSCHITZ")


if __name__ == "__main__":
    main()
