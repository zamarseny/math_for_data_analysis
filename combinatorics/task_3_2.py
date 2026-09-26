"""
B. Конфигурации с трёхпозиционными переключателями
Ограничение времени
2 с
Ограничение памяти
256.0 Мб
Ввод
стандартный ввод
Вывод
стандартный вывод
Есть n  независимых переключателей, каждый находится в одном из трёх состояний: OFF, WEAK, STRONG. Требуется посчитать число конфигураций, в которых встречается хотя бы один переключатель в состоянии WEAK и хотя бы один переключатель в состоянии STRONG.

Формат входных данных
Одно целое n: (0≤n≤10**18).

Формат выходных данных
Одно число — ответ по модулю 1_000_000_007.

Пример
Ввод
2

Вывод
2
"""

import sys
import math


def to_n_decimal(numb: int, n: int, le: int) -> str:
    """
    n<20 - n-decimal, if n==2 - binary number system
    for lexicographic order
    l - lenght with leading zeros
    """
    figs = [str(i) for i in range(10)]
    figs.extend([chr(i) for i in range(65, 91)])
    figs = figs[:n]
    quotient = numb
    residual = quotient % n
    st = ""
    while quotient >= n:
        residual = quotient % n
        quotient = quotient // n
        st = str(figs[residual]) + st
        # print(f"quot: {quotient}, resid: {residual}, cur_st: {st}")
    if quotient > 0:
        st = str(figs[quotient]) + st
    # leading zeros
    if len(st) < le:
        st = "0" * (le - len(st)) + st
    return st


def calc_combs__w_lim(inp_n: int) -> int:
    combs_num = 0
    necc_len = inp_n
    for i in range(inp_n**3):
        # necc_len = math.ceil(math.log(inp_n, 3))
        # print(f"necc_len: {necc_len}")
        cur_comb = to_n_decimal(numb=i, n=3, le=necc_len)
        # print(cur_comb)
        if "0" in cur_comb and "2" in cur_comb:
            combs_num += 1
    return combs_num


def calc_combs_incl(n: int):
    combs = 3**n - 2 * 2**n + 1
    return combs


def main():
    """
    Пример ввода и вывода числа n, где -10^9 < n < 10^9:
    n = int(input())
    print(n)
    """
    n = int(input())
    # combs = calc_combs__w_lim(inp_n=n)
    # combs = 3 ** (n - 2) * n * (n - 1)
    combs = calc_combs_incl(n=n)
    print(combs)


if __name__ == "__main__":
    main()
    # st = to_n_decimal(numb=13, n=13, le=3)
    # print(st)

    # resp = calc_combs__w_lim(inp_n=3)
    # print(resp)
    # for i in range(27):
    #     st = to_n_decimal(numb=i, n=3, le=3)
    #     print(st)
