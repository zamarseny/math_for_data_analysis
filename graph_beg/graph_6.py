import sys


def main():
    """
    is full graph
    """
    with open("input.txt", "r") as f_in:
        users = f_in.readline().split()
        movies = f_in.readline().split()
        viewings = f_in.readlines()

    if len(viewings) > 0:
        answer = "YES"
    else:
        answer = "NO"

    for viewing in viewings:
        met_us, num, *items = viewing.split()
        print(met_us, num, items)
        for movie in movies:
            if movie not in items:
                answer = "NO"
                break
        if answer == "NO":
            break

    print(answer)

    with open("output.txt", "w") as f_out:
        f_out.write(answer)


if __name__ == "__main__":
    main()
