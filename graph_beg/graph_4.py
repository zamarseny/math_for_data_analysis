import sys
import numpy as np


def main():
    """
    matrix multiplication - is bad idea. Complexity is O(n^4).
    """
    adj_matr = []
    with open("input.txt") as file:
        rows = file.readlines()
        size = int(rows[0])
        for i, row in enumerate(rows[1:-1]):
            row = row.split()
            row = [int(x) for x in row]
            adj_matr.append(row)
    from_pos, to_pos = int(rows[-1].split()[0]), int(rows[-1].split()[1])
    adj_matr = np.array(adj_matr)
    min_path_len = 0
    min_path_found = False

    while not min_path_found:
        min_path_len += 1
        pow = np.linalg.matrix_power(adj_matr, min_path_len)
        if pow[from_pos][to_pos] > 0:
            min_path_found = True
            break
        if np.linalg.matrix_rank(pow) == 0:
            min_path_len = -1
            break
    print(min_path_len)

    with open("output.txt", "w") as file:
        file.write(str(min_path_len))


if __name__ == "__main__":
    main()
