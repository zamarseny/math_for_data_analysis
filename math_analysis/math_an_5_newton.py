def load_func_coefficients():
    """
    1 -2 -3
    4
    1e-6
    """
    with open("input.txt", "r") as f_in:
        a, b, c = [float(i) for i in f_in.readline().split()]
        x_0 = float(f_in.readline())
        eps = float(f_in.readline())
    return a, b, c, x_0, eps


def calc_value(a, b, c, x):
    return a * x**2 + b * x + c


def first_derivative(a, b, c, x):
    return 2 * a * x + b


def find_root_newton(a, b, c, x_0, eps):
    root = x_0
    iteration = 0
    solution_found = False
    while abs(calc_value(a, b, c, root)) > eps:
        if abs(first_derivative(a, b, c, root)) < eps or iteration > 1000:
            return root, iteration, solution_found
        root -= calc_value(a, b, c, root) / first_derivative(a, b, c, root)
        iteration += 1
    solution_found = True
    return root, iteration, solution_found


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    a, b, c, x_0, eps = load_func_coefficients()

    print(a, b, c, x_0, eps)

    root, iteration, solution_found = find_root_newton(a, b, c, x_0, eps)

    with open("output.txt", "w") as f_out:
        if solution_found:
            output_text = f"Root found: x = {root:.6f}" + "\n"
            output_text += f"Number of iterations: {iteration}"
            f_out.write(str(output_text))
        else:
            f_out.write("Solution not found")


if __name__ == "__main__":
    main()
