def load_graph():
    """ """

    edges = []
    with open("input.txt", "r") as f_in:
        n_vertices = int(f_in.readline())
        vertices = f_in.readline().split()
        num_edges = int(f_in.readline())
        for i in range(num_edges):
            from_pos, to_pos = f_in.readline().split()
            edges.append([from_pos, to_pos])
    return n_vertices, vertices, num_edges, edges


def is_theorem_true(edges):
    from_positions = [i[0] for i in edges]
    to_positions = [i[1] for i in edges]
    print(from_positions, to_positions)
    for to_pos in to_positions:
        if to_pos not in from_positions:
            return True
    return False


def main():
    """ """
    n_vertices, vertices, num_edges, edges = load_graph()
    print(n_vertices, vertices, num_edges, edges)

    is_theorem_tr = is_theorem_true(edges) and len(edges) > 0
    print(is_theorem_tr)
    if not is_theorem_tr:
        with open("output.txt", "w") as f_out:
            f_out.write("NO")
    else:
        with open("output.txt", "w") as f_out:
            f_out.write("YES")


if __name__ == "__main__":
    main()
