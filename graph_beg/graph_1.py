import sys


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    n_loops = 0
    loops = []
    with open("input.txt") as file:
        rows = file.readlines()
        for i, row in enumerate(rows):
            row = row.split()
            row = [int(x) for x in row]
            if row[i] > 0:
                n_loops += 1
                loops.append(str(i))
    if n_loops == 0:
        loops = "NO LOOPS"
        with open("output.txt", "w") as file:
            file.write(loops)
    else:
        # loops = ", ".join(loops)
        with open("output.txt", "w") as file:
            for i in range(len(loops)):
                file.write(str(loops[i]))
                file.write("\n")


if __name__ == "__main__":
    main()
