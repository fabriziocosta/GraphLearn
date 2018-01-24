from collections import defaultdict


def _extract_grammar_stats(grammar):
    count_corehashes = defaultdict(int)
    count_interfacehashes = defaultdict(int)
    corecounter = defaultdict(int)
    intercounter = defaultdict(int)
    for ih in grammar.keys():
        for ch in grammar[ih].keys():
            # go over all the combos
            count_corehashes[ch] += 1
            count_interfacehashes[ih] += 1
            count = grammar[ih][ch].count
            corecounter[ch] += count
            intercounter[ih] += count
    return count_corehashes, count_interfacehashes, corecounter, intercounter


import networkx as nx
import graphlearn.lsgg as lsgg


def _edenize_for_testing(g):
    for n in g.nodes():
        g.node[n]['label'] = str(n)
    for a, b in g.edges():
        g[a][b]['label'] = '.'
    return g


def test_get_grammar():
    lsggg = lsgg.lsgg()
    g = _edenize_for_testing(nx.path_graph(4))
    lsggg.fit([g, g, g])
    return lsggg

def test_dark_edges():
    ''''''
    lsggg = lsgg.lsgg()
    g = _edenize_for_testing(nx.path_graph(3))

    g_dark = g.copy()
    for n,d in g_dark.nodes(data=True):
        if 'edge' in d:
            d['nesting'] = True


    # test fit
    lsggg.fit([g, g_dark])
    assert 2 == len([g.graph  for v in lsggg.productions.values() for g in v.values()])

    # test production
    res = lsggg.neighbors(g_dark)

    #import structout as so
    #so.gprint([g.graph  for v in lsggg.productions.values() for g in v.values()] )
    assert 2 == len(list(res))

def test_get_circular_graph():
    G=nx.path_graph(8)
    G.add_edge(7,3)
    G.add_edge(1,3)
    G=_edenize_for_testing(G)
    return G

def valid_gl_graph(graph):
    """checks if a graph is a valid graphlearn-intermediary product"""

    secondlayer = 'original' in graph.graph

    # are labels in the graph?
    def label_is_set(graph):
        for a,b,d in graph.edges(data=True):
            if 'label' not in d: return False
        for n,d in graph.nodes(data=True):
            if 'label' not in d: return False
        return True

    assert label_is_set(graph)
    if secondlayer:
        assert label_is_set(graph.graph['original']) == True

    # second layer needs contracted attributes...
    if secondlayer:
         assert set(graph.graph['original'].nodes()) == reduce(lambda a,b: a|b , [ d['contracted'] for n,d in graph.nodes(data=True)] )

    return True


