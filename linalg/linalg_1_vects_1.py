"""
input:

3
1.5 -2 0.5
1 2 3
4 5 6
7 8 9

exp out:
-3.0 -3.0 -3.0


Example 2

input:
1
-1
3 5 7 9

Output:
-3.0 -5.0 -7.0 -9.0
"""

import sys


class Vector:
    def __init__(self, coeffs: array):
        self.coeffs = coeffs
        self.dim = len(coeffs)

    def __add__(self, another_vect: Vector):
        self.coeffs = [
            (self.coeffs[i] + another_vect.coeffs[i]) for i in range(self.dim)
        ]
        return self

    def const_mult(self, const):
        self.coeffs = [i * const for i in self.coeffs]
        return self

    def __str__(self):
        return " ".join(str(i) for i in self.coeffs)


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    k = int(input())
    coeffs = [float(i) for i in input().split()]

    # Read first vector to determine dimension
    first_vect = Vector([float(j) for j in input().split()])
    # lin_comb = Vector([0.0 for i in range(len(coeffs))])
    lin_comb = Vector([0.0 for i in range(first_vect.dim)])
    lin_comb = lin_comb.__add__(another_vect=first_vect.const_mult(coeffs[0]))

    # Process remaining vectors
    for i in range(1, k):
        vect = Vector([float(j) for j in input().split()])
        lin_comb = lin_comb.__add__(another_vect=vect.const_mult(coeffs[i]))
    # print(lin_comb.coeffs)
    print(lin_comb.__str__())


if __name__ == "__main__":
    main()
