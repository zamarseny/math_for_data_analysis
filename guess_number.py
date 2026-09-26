def guess_number(lower_bound: int, upper_bound: int, number: int, n_tries: int) -> int:
    """
    tries to guess the number between upper_bound and lower_bound.
    """
    n_tries += 1
    mid = (upper_bound + lower_bound) // 2
    if number == mid:
        return mid, n_tries
    elif number > mid:
        return guess_number(mid + 1, upper_bound, number, n_tries + 1)
    else:
        return guess_number(lower_bound, mid - 1, number, n_tries + 1)


if __name__ == "__main__":
    print(guess_number(lower_bound=1, upper_bound=100, number=75, n_tries=0))
