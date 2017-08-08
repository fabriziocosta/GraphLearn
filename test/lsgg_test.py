

import networkx as nx
import graphlearn.lsgg as lsgg
import graphlearn.lsgg_cip as lcu


def edenize(g):
    for n in g.nodes():
        g.node[n]['label']=str(n)
    for a,b in g.edges():
        g[a][b]['label']='.'
    return g

def prep_cip_extract(g):
    g= edenize(g)
    return g

def get_grammar():
    lsggg=lsgg.lsgg()
    g=prep_cip_extract(nx.path_graph(4))
    lsggg.fit([g,g,g])
    return lsggg

def test_fit():
    lsggg= get_grammar()

    assert( 4 == sum( len(e)  for e in lsggg.productions.values()) )
    assert(32346 in lsggg.productions[1354])
    assert(55730 in lsggg.productions[1354])
    assert(39282 in lsggg.productions[31148])
    assert(30237 in lsggg.productions[31148])
    #gprint( [e.graph for e in lsggg.productions[49532].values() ])
    #gprint( [e.graph for e in lsggg.productions[29902].values() ])


def test_extract_core_and_interface():
    graph=nx.path_graph(4)
    prep_cip_extract(graph)
    res = lcu.extract_core_and_interface(root_node=3, graph=graph, radius=1,thickness=1)
    #gprint(res.graph)
    assert ( str(res) == "cip: int:16931, cor:352004, rad:1, thi:1, rot:3")


def test_neighbors():
    # make a grammar
    lsggg = get_grammar()

    #make agraph
    g=nx.path_graph(4)
    g=edenize(g)
    g.node[3]['label']='5'
    stuff=list(lsggg.neighbors(g))
    assert(6 ==  len(stuff))

def test_some_neighbors():
    # make a grammar
    lsggg = get_grammar()
    #make agraph
    g=nx.path_graph(4)
    g=edenize(g)
    g.node[3]['label']='5'
    assert ( 1== len(list( lsggg.neighbors_sample(g,1) )))
    assert ( 2== len(list( lsggg.neighbors_sample(g,2) )))
    assert ( 3== len(list( lsggg.neighbors_sample(g,3) )))
    #gprint(list( lsggg.some_neighbors(g,1) ))
    #gprint(list( lsggg.some_neighbors(g,2) ))
    #gprint(list( lsggg.some_neighbors(g,3) ))





if __name__=="__main__":
    test_fit()
    test_extract_core_and_interface()
    test_neighbors()
    test_some_neighbors()
