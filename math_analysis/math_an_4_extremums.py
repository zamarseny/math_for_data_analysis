import math


def load_func_coefficients():
    with open("input.txt", "r") as f_in:
        a, b, c, d = [float(i) for i in f_in.readline().split()]
        p, q = [float(i) for i in f_in.readline().split()]
    return a, b, c, d, p, q


def calc_value(a, b, c, d, x):
    return a * x**3 + b * x**2 + c * x + d


def first_derivative(a, b, c, d, x):
    return 3 * a * x**2 + 2 * b * x + c


def second_derivative(a, b, c, d, x):
    return 6 * a * x + 2 * b


def find_extremums(a, b, c, d, p, q):
    D = b**2 - 3 * a * c
    if a == 0 and b != 0:
        x1 = -c / (2 * b)
        x2 = x1
    elif a == 0 and b == 0:
        x1 = -1 * 10**5
        x2 = -1 * 10**5
    else:
        if D > 0:
            x1 = (-b - math.sqrt(D)) / (3 * a)
            x2 = (-b + math.sqrt(D)) / (3 * a)
        elif D == 0:
            x1 = -b / (3 * a)
            x2 = x1
        else:
            x1 = -1 * 10**5
            x2 = -1 * 10**5
        return x1, x2


def exremum_type(a, b, c, d, x):
    if second_derivative(a, b, c, d, x) > 0:
        return "Local minimum"
    elif second_derivative(a, b, c, d, x) < 0:
        return "Local maximum"
    else:
        return "Saddle point"


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    a, b, c, d, p, q = load_func_coefficients()

    x1, x2 = find_extremums(a, b, c, d, p, q)

    with open("output.txt", "w") as f_out:
        output_text = ""
        if abs(x1 - x2) < 1e-6:
            potential_extremums = [x1]
        else:
            potential_extremums = [x1, x2]
        for x in potential_extremums:
            if x >= p and x <= q:
                output_text += f"{exremum_type(a, b, c, d, x)} at x = {x:.5f}\n"
                output_text += f"f(x) = {calc_value(a, b, c, d, x):.5f}\n"
        if output_text == "":
            output_text = "No critical points found."
        f_out.write(output_text)


if __name__ == "__main__":
    main()
