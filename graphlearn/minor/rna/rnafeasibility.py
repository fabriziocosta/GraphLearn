'''
rna feasibility checker
'''
import networkx as nx

def is_rna(graph):
    graph = graph.copy()
    # remove structure
    bonds = [n for n, d in graph.nodes(data=True) if d['label'] == '=']
    graph.remove_nodes_from(bonds)
    # see if we are cyclic
    for node, degree in graph.in_degree_iter(graph.nodes()):
        if degree == 0:
            break
    else:
        return False
    # check if we are connected.
    graph = nx.Graph(graph)
    return nx.is_connected(graph)