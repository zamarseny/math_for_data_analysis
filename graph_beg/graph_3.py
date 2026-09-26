import sys


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    adj_matr = []
    with open("input.txt") as file:
        rows = file.readlines()
        lines_n = int(rows[0])
        for i, row in enumerate(rows[1:]):
            row_inds = row.split()
            row_inds = [int(x) for x in row_inds]
            i_adj = [1 if i in row_inds else 0 for i in range(lines_n)]
            adj_matr.append(i_adj)

    with open("output.txt", "w") as file:
        for line in adj_matr:
            file.write(" ".join([str(i) for i in line]))
            file.write("\n")


if __name__ == "__main__":
    main()
