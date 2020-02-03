#from graphlearn 
import lsgg_pisi
import json 
import networkx as nx 
import structout as so
import random
def getgraphs():
    with open("test/chemtest.json", 'r') as handle:                                      
          stuff = json.loads(handle.read())
          def js_to_graph(x):
              g= nx.readwrite.node_link_graph(x)
              g.cip_graph={}
              return g
          return  [js_to_graph(x) for x in stuff]



def test_pisi():
    g= getgraphs()
    grammar = lsgg_pisi.PiSi(  
            decomposition_args={"radius_list": [0,1], 
                                "thickness": 1,  
                                "pisi_minsimilarity": .9,  # doesnt exist anymore oO ?
                                "thickness_pisi": 4},
            filter_args={"min_cip_count": 1,                               
                         "min_interface_count": 1}
            )
    #print(g[:10])
    grammar.fit(g[:100])
    '''
    for k,v in grammar.productions.items():
        for kk, vv in v.items():
            print(len(vv.pisi_vectors),vv.count)
    '''
    print("fitting done")
    neighs = list(grammar.neighbors(g[44]))
    so.gprint(neighs [:3])
    print ("numnei:", len(neighs))


#test_pisi()
