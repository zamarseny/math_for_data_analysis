import sys


def main():
    """ """
    with open("input.txt", "r") as f_in:
        n = int(f_in.readline())
        matrix = f_in.readlines()

    leaves = []
    for i, row in enumerate(matrix):
        row = row.split()
        row = [int(x) for x in row]

        vert_pow = 0
        for j, el in enumerate(row):
            if el > 0:
                vert_pow += 1
        if vert_pow == 1:
            leaves.append(i)

    print(leaves)

    with open("output.txt", "w") as f_out:
        f_out.write("\n".join([str(i) for i in leaves]))


if __name__ == "__main__":
    main()
