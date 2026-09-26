import sys


def load_graph():
    """
    5
    0 1 0 0 0
    1 0 1 1 0
    0 1 0 0 0
    0 1 0 0 1
    0 0 0 1 0
    1 4
    #final output: 1 3 4

    5
    0 1 0 0 0
    1 0 1 1 0
    0 1 0 0 0
    0 1 0 0 1
    0 0 0 1 0
    1 4
    #final output: NO PATH
    """

    matrix = []
    with open("input.txt", "r") as f_in:
        n = int(f_in.readline())
        for i in range(n):
            row = f_in.readline().split()
            row = [int(x) for x in row]
            matrix.append(row)
        from_pos, to_pos = f_in.readline().split()
    return n, int(from_pos), int(to_pos), matrix


def adj_matr_to_edges(matrix):
    edges = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > 0:
                edges.append((i, j, matrix[i][j]))
    return edges


def find_shortest_way(vertices, edges, root_pos, final_pos):
    shortest_dists = [float("inf") for i in range(len(vertices))]
    shortest_dists[root_pos] = 0
    shortest_ways = [[root_pos] for i in range(len(vertices))]
    uncovered_vertices = set(vertices)  # use set for O(1) remove/lookup
    close_shortest_inds = [root_pos]  # frontier: vertices to process this round
    step_no = 0
    while len(uncovered_vertices) > 0:
        len_before = len(uncovered_vertices)
        next_frontier = []  # vertices reached this round for next iteration
        for cur_vert in close_shortest_inds:
            cur_edges = [
                i for i in edges if i[0] == cur_vert and i[1] in uncovered_vertices
            ]
            for cur_edge in cur_edges:
                to_pos, weight = cur_edge[1:]
                if shortest_dists[to_pos] > (shortest_dists[cur_vert] + weight):
                    shortest_dists[to_pos] = shortest_dists[cur_vert] + weight
                    shortest_ways[to_pos] = shortest_ways[cur_vert] + [to_pos]
                next_frontier.append(to_pos)
            if cur_vert in uncovered_vertices:
                uncovered_vertices.remove(cur_vert)
            step_no += 1
        # No progress: remaining uncovered vertices are unreachable (e.g. other component)
        if len(uncovered_vertices) == len_before and len(uncovered_vertices) > 0:
            break
        close_shortest_inds = list(set(next_frontier))
        if not close_shortest_inds:
            break

    # No path if final vertex is unreachable
    if shortest_dists[final_pos] == float("inf"):
        return None
    return shortest_ways[final_pos]


def main():
    """ """
    n, root_pos, final_pos, matrix = load_graph()
    vertices = list(range(n))
    print(n, root_pos, final_pos, matrix)
    edges = adj_matr_to_edges(matrix)
    print(edges)
    shortest_way = find_shortest_way(vertices, edges, root_pos, final_pos)

    if shortest_way is None:
        print(-1)
        with open("output.txt", "w") as f_out:
            f_out.write("NO PATH")
    else:
        print(shortest_way)
        with open("output.txt", "w") as f_out:
            f_out.write(" ".join([str(i) for i in shortest_way]))


if __name__ == "__main__":
    main()
