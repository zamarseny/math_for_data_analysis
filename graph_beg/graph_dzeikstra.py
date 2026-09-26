import sys
import networkx as nx
import numpy as np


def load_graph():
    # wikipedia example
    # edges = [
    #     (1, 2, 7),
    #     (1, 3, 9),
    #     (1, 6, 14),
    #     (2, 4, 15),
    #     (2, 3, 10),
    #     (3, 4, 11),
    #     (3, 6, 2),
    #     (4, 5, 6),
    #     (5, 6, 9),
    # ]
    # yandex example
    edges = [
        (1, 2, 1),
        (1, 3, 4),
        (2, 3, 2),
        (2, 4, 5),
        (3, 4, 1),
    ]
    edges.extend([(i[1], i[0], i[2]) for i in edges])

    edges = [[i[0] - 1, i[1] - 1, i[2]] for i in edges]
    vertices = set()
    for edge in edges:
        vertices.add(edge[0])
        vertices.add(edge[1])
    vertices = list(vertices)
    return edges, vertices


def dijkstra(edges, vertices, start_vertex):
    shortest_ways = [np.inf for i in range(len(vertices))]
    shortest_ways[start_vertex] = 0
    uncovered_vertices = vertices.copy()
    cur_vert = start_vertex
    step_no = 0
    while len(uncovered_vertices) > 0:
        # calc distances
        cur_edges = [
            i for i in edges if i[0] == cur_vert and i[1] in uncovered_vertices
        ]
        # print("step", step_no, "cur_vert", cur_vert)
        # print("cur_edges", cur_edges)
        step_dists = [i[2] for i in cur_edges]  # cur_edges[:,2]

        # check lowest
        close_shortest_dis = np.inf

        # dummy initialization
        # close_shortest_ind = cur_edges[0][1]
        for cur_edge in cur_edges:
            to_pos, weight = cur_edge[1:]

            # update close vert dists
            if shortest_ways[to_pos] > (shortest_ways[cur_vert] + weight):
                shortest_ways[to_pos] = shortest_ways[cur_vert] + weight
            # next vert
            if close_shortest_dis > weight:
                close_shortest_dis = weight
                close_shortest_ind = to_pos
        # print("uncovered_vertices", uncovered_vertices)
        uncovered_vertices.remove(cur_vert)
        cur_vert = close_shortest_ind
        # print("closest_ind", close_shortest_ind)
        # print("close_shortest_dis", close_shortest_dis)
        # print("\n")

        step_no += 1

    return shortest_ways


def main():
    """ """
    edges, vertices = load_graph()
    print(edges, vertices)
    start_vertex = 0
    shortest_ways = dijkstra(edges, vertices, start_vertex)
    print(shortest_ways)


if __name__ == "__main__":
    main()
