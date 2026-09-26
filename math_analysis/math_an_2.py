def calc_value(text_f, x):
    return eval(text_f)


def is_continuous(text_f, x_0, sigma, eps) -> bool:
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
    k = 5
    with open("input.txt", "r") as f_in:
        text_f = f_in.readline()
        x_0 = float(f_in.readline())
        sigma = float(f_in.readline())
    eps = k * sigma

    # print(text_f, x_0, sigma, eval(text_f))

    is_contin = is_continuous(text_f, x_0, sigma, eps)

    # print(is_contin)

    with open("output.txt", "w") as f_out:
        if is_contin:
            f_out.write("CONTINUOUS")
        else:
            f_out.write("DISCONTINUOUS")


if __name__ == "__main__":
    main()
