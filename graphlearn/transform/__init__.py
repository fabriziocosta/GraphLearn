



def merge_edge(graph, u, v):

    new_edges = ((u, w, d) for x, w, d in graph.edges([v], data=True)
                     if w != u)

    graph.remove_node(v)
    graph.add_edges_from(new_edges)