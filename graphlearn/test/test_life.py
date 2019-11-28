#from graphlearn 
import lsgg_life 
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



def test_life():
    g= getgraphs()
    grammar = lsgg_life.LIFE(  
            decomposition_args={"radius_list": [0,1], 
                                "thickness_list": [1],  
                                "life_minsimilarity": .9, 
                                "thickness_life": 4},
            filter_args={"min_cip_count": 1,                               
                         "min_interface_count": 1}
            )
    #print(g[:10])
    grammar.fit(g[:100])
    print("fitting done")
    neighs = list(grammar.neighbors(g[55]))
    so.gprint(neighs [:3])
    print ("numnei:", len(neighs))


#test_life()
