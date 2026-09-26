import sys
import itertools


def load_data(f_in):
    """
    4 2
    1 1 3 3
    """

    n, batch_size = [int(i) for i in f_in.readline().split()]
    grad_vals = [float(i) for i in f_in.readline().split()]

    return n, batch_size, grad_vals


def main():
    n, batch_size, grad_vals = load_data(sys.stdin)
    # print(n, batch_size, grad_vals)
    batches = list(itertools.combinations(grad_vals, batch_size))

    # mean_batches = (
    #     sum(v for v in batch for batch in batches) / len(batches) / batch_size
    # )
    # var_batches = (
    #     sum((v - mean_batches) ** 2 for v in batch for batch in batches)
    #     / len(batches)
    #     / batch_size
    # )
    # print(list(itertools.combinations(grad_vals, batch_size)))
    grad_mean = sum(grad_vals) / n
    mean = 0.0
    variance = 0.0
    for i, batch in enumerate(batches):
        mean_batch = sum(v for v in batch) / batch_size
        mean += mean_batch / len(batches)
        var_batch = (
            grad_mean - mean_batch
        )  # sum((v - mean_batch) ** 2 for v in batch) / batch_size
        variance += var_batch**2 / len(batches)
        # print(i, batch, mean_batch, var_batch)

    #
    # variance = sum((grad_val - mean) ** 2 for grad_val in grad_vals) / n
    print(f"{mean:.6f} {variance:.6f}")


if __name__ == "__main__":
    main()
