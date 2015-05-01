import networkx as nx
from eden.graph import Vectorizer
import random
'''
    wrapped or altered eden functions:
        -my_vectorizer, a eden vectorizer that doesnt try to expand graphs
        -my_fit_estimator, a estimator that doesnt do validation and doesnt print 
        -expand_edges, just for convenience.
        
'''


class GraphLearnVectorizer(Vectorizer):
    '''
    doing some overwriting so we dont expand and contract edges all the time..
    this hack is a little bit dependant on the state of eden.. so be carefull here 
    '''

    def transform2(self, graph):
        G=graph.copy()
        return self._convert_dict_to_sparse_matrix(self._transform(0, G))


    # ok so our vectorizer should be expected to work with already expanded graphs...
    def _edge_to_vertex_transform(self, original_graph):
        """Converts edges to nodes so to process the graph ignoring the information on the
        resulting edges."""

        if 'expanded' in original_graph.graph:
            return original_graph

        G = nx.Graph()
        G.graph['expanded']=True
        # build a graph that has as vertices the original vertex set
        for n, d in original_graph.nodes_iter(data=True):
            d['node'] = True
            G.add_node(n, d)
        # and in addition a vertex for each edge
        new_node_id = max(original_graph.nodes()) + 1
        for u, v, d in original_graph.edges_iter(data=True):
            d['edge'] = True
            G.add_node(new_node_id, d)
            # and the corresponding edges
            G.add_edge(new_node_id, u, label=None)
            G.add_edge(new_node_id, v, label=None)
            new_node_id += 1
        return G



def expand_edges(graph):
    '''
    convenience wrapper
    '''
    vectorizer = Vectorizer(complexity= 3)
    return vectorizer._edge_to_vertex_transform(graph)


def contract_edges(original_graph):
    """
        stealing from eden...
        because i draw cores and interfaces there may be edge-nodes 
        that have no partner, eden gives error in this case.
        i still want to see them :) 
    """
    # start from a copy of the original graph
    G = nx.Graph(original_graph)
    # re-wire the endpoints of edge-vertices
    for n, d in original_graph.nodes_iter(data=True):
        if d.get('edge', False) == True:
            # extract the endpoints
            endpoints = [u for u in original_graph.neighbors(n)]
            # assert (len(endpoints) == 2), 'ERROR: more than 2 endpoints'
            if len(endpoints) != 2:
                continue
            u = endpoints[0]
            v = endpoints[1]
            # add the corresponding edge
            G.add_edge(u, v, d)
            # remove the edge-vertex
            G.remove_node(n)
        if d.get('node', False) == True:
            # remove stale information
            G.node[n].pop('remote_neighbours', None)
    return G



def select_random(graph_iter, len,samplesize):
    x=range(len)
    random.shuffle(x)
    x=x[:samplesize]
    x.sort(reverse=True)
    next=x.pop()
    for i,g in enumerate(graph_iter):
        if i==next:
            yield g
            if not x:
                break
            next=x.pop()

from collections import defaultdict

def calc_stats_from_grammar(grammar):
    count_corehashes = defaultdict(int)
    count_interfacehashes = defaultdict(int)
    corecounter = defaultdict(int)
    intercounter = defaultdict(int)
    for ih in grammar.keys():
        for ch in grammar[ih].keys():
            # go over all the combos
            count_corehashes[ch]+=1
            count_interfacehashes[ih]+=1
            count= grammar[ih][ch].count
            corecounter[ch]+=count
            intercounter[ih]+=count
    return count_corehashes,count_interfacehashes,corecounter,intercounter






