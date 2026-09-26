import sys


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    adj_list = []
    with open("input.txt") as file:
        rows = file.readlines()
        for i, row in enumerate(rows):
            row = row.split()
            row = [int(x) for x in row]
            i_adj = []
            for j in range(len(row)):
                if row[j] > 0:
                    i_adj.append(str(j))

            i_adj += "\n"
            adj_list.append(i_adj)
    # print(adj_list)
    with open("output.txt", "w") as file:
        for line in adj_list:
            file.write(" ".join(line))
            # file.write("\n")


if __name__ == "__main__":
    main()
