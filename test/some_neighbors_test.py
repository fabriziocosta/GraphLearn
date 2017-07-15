import networkx as nx
from graphlearn.extensions_lsgg.some_neighbors import lsgg


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
    lsggg=lsgg()
    g=prep_cip_extract(nx.path_graph(4))
    lsggg.fit([g,g,g])
    return lsggg

def test_some_neighbors():
    # make a grammar
    lsggg = get_grammar()
    #make agraph
    g=nx.path_graph(4)
    g=edenize(g)
    g.node[3]['label']='5'
    assert ( 1== len(list( lsggg.some_neighbors(g,1) )))
    assert ( 2== len(list( lsggg.some_neighbors(g,2) )))
    assert ( 3== len(list( lsggg.some_neighbors(g,3) )))
    #gprint(list( lsggg.some_neighbors(g,1) ))
    #gprint(list( lsggg.some_neighbors(g,2) ))
    #gprint(list( lsggg.some_neighbors(g,3) ))





if __name__=="__main__":
    test_some_neighbors()

