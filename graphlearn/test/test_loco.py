#from graphlearn 
import lsgg_loco
import json 
import networkx as nx 
import structout as so
import random
def getgraphs():
    with open("test/chemtest.json", 'r') as handle:                                      
          stuff = json.loads(handle.read())
          def js_to_graph(x):
              g= nx.readwrite.node_link_graph(x)
              g.graph={}
              return g
          return  [js_to_graph(x) for x in stuff]



def test_loco():
    g= getgraphs()
    grammar = lsgg_loco.LOCO(  
            decomposition_args={"radius_list": [0,1], 
                                "thickness_list": [1],  
                                "loco_minsimilarity": .9, 
                                "thickness_loco": 4},
            filter_args={"min_cip_count": 1,                               
                         "min_interface_count": 1}
            )
    #print(g[:10])
    grammar.fit(g[:100])
    '''
    for k,v in grammar.productions.items():
        for kk, vv in v.items():
            print(len(vv.loco_vectors),vv.count)
    '''
    print("fitting done")
    neighs = list(grammar.neighbors(g[44]))
    so.gprint(neighs [:3])
    print ("numnei:", len(neighs))


#test_loco()
