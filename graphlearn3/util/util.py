from collections import defaultdict
import functools
import networkx as nx
from graphlearn3 import lsgg

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

def get_cyclegraphs():
    g1 = nx.path_graph(5)
    g1.add_edge(2,4)

    g2 = nx.path_graph(7)
    g2.add_edge(1,3)
    g2.add_edge(5,3)

    g3 = nx.path_graph(3)
    g3.add_edge(4,3)
    g3.add_edge(4,2)
    g3.add_edge(4,1)
    
    G = [g1,g2,g3]
    G = list(map(_edenize_for_testing,G))
    G.append(test_get_circular_graph())
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
         assert set(graph.graph['original'].nodes()) == functools.reduce(lambda a,b: a|b , [ d['contracted'] for n,d in graph.nodes(data=True)] )

    return True



def decorate_cip(cip):
    #print (cip.core_nodes)
    #print (cip.interface_nodes)
    nx.set_node_attributes(cip.graph,{n:True for n in cip.core_nodes} ,'core')
    nx.set_node_attributes( cip.graph,{n:True for n in cip.interface_nodes} ,'interface')
    print ('decoration not needed anymore,  pass nodecolorgrouplists to gprint')



def draw_grammar_term(grammarobject,
                       n_productions=10,
                     n_graphs_per_line=5,
                     n_graphs_per_production=5,
                     size=10):

    import structout as so

    print (str(grammarobject)+ "          | cores are cyan")

    grammar = grammarobject.productions
    if n_productions is None or len(grammar) < n_productions:
        n_productions = len(grammar)

    most_prolific_productions = sorted(
        [(len(grammar[interface]), interface) for interface in grammar],
        key=lambda x: x[0],
        reverse=True)

    for i in range(n_productions):
        interface = most_prolific_productions[i][1]
        core_cid_dict = grammar[interface]

        cips = [core_cid_dict[chash] for chash in core_cid_dict.keys()]

        # list(map(decorate_cip, cips)) see color argument

        most_frequent_cips = sorted([(cip.count, cip) for cip in cips], reverse=True, key=lambda x:x[0])
        graphs = [cip.graph for count, cip in most_frequent_cips]
        color = [(cip.interface_nodes,cip.core_nodes) for count, cip in most_frequent_cips]
        # graphs =[cip.abstract_view for count, cip in most_frequent_cips]


        graphs = graphs[:n_graphs_per_production]
        # dists = [core_cid_dict[chash].distance_dict for i, chash in enumerate(core_cid_dict.keys()) \
        # if i < 5]
        print('interface id: %s [%d options]' % (interface, len(grammar[interface])))


        so.gprint(graphs,
                  color= color,
                   n_graphs_per_line=n_graphs_per_line,
                   size=size)

