

class no_transform:
    def encode(self, thing):
        return thing
    def decode(self, thing):
        return thing

def merge_edge(graph, u, v):
    new_edges = ((u, w, d) for x, w, d in list(graph.edges.data(nbunch=v)) if w != u)
    #new_edges = ((u, w, d) for x, w, d in graph.edges([v], data=True) if w != u)

    graph.remove_node(v)
    graph.add_edges_from(new_edges)