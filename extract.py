import networkx as nx
from eden import fast_hash
'''
the interesting things here are:

extract_core_and_interface(node,graph,radius[],thickness[])
    we create "core_interface_data" for all radius/thickness pairs

preprocess(graph)
    before extracting core/interface a graph needs to be prepared.
    
answer 
    saves info on one extracted core / interface pair
'''


class core_interface_data:
    def __init__(self, ihash, chash, graph, radius, thickness):
        self.interface_hash = ihash
        self.core_hash = chash
        self.graph = graph
        self.radius = radius
        self.thickness = thickness


bitmask = 2 ** 20 - 1




def inversedict(d):
    d2 = {}
    for k, v in d.iteritems():
        d2[v] = d2.get(v, [])
        d2[v].append(k)
    return d2


def calc_interface_hash(interfacegraph):
    l = []
    node_name_cache = {}
    for (a, b) in interfacegraph.edges():

        ha = node_name_cache.get(a, -1)
        if ha == -1:
            ha = calc_node_name(interfacegraph, a)
            node_name_cache[a] = ha
        hb = node_name_cache.get(b, -1)
        if hb == -1:
            hb = calc_node_name(interfacegraph, b)
            node_name_cache[b] = hb
        l.append((ha ^ hb) + (ha + hb))
    l.sort()
    l = fast_hash(l,bitmask)
    return l


def calc_node_name(interfacegraph, node):
    d = nx.single_source_shortest_path_length(interfacegraph, node, 20)
    # d is now node:dist
    # l is a list of  hash(label,distance)
    #l=[   func([interfacegraph.node[nid]['intlabel'],dis])  for nid,dis in d.items()]
    l = [interfacegraph.node[nid]['hlabel'][0] + dis for nid, dis in d.items()]
    l.sort()
    l = fast_hash(l,bitmask)
    return l


def extract_core_and_interface(node, graph, radius, thickness):
    # which nodes are in the relevant radius
    dist = nx.single_source_shortest_path_length(graph, node, max(radius) + max(thickness))
    # we want the relevant subgraph and we want to work on a copy
    retgraph = nx.Graph(graph.subgraph(dist))

    # we want to inverse the dictionary.
    # so now we see {distance:[list of nodes at that distance]}
    nodedict = inversedict(dist)

    #sanity check.. if this doesnt exist we couldnt create anything new.. so we just default
    if max(radius) + max(thickness) not in nodedict:
        return []

    retlist = []
    for thickness_ in thickness:
        for radius_ in radius:

            #calculate hashes
            #d={1:[1,2,3],2:[3,4,5]}
            #print [ i for x in [1,2] for i in d[x] ]
            interface_graph_nodes = [item for x in range(radius_ + 1, radius_ + thickness_ + 1) for item in
                                     nodedict.get(x, [])]
            interfacehash = calc_interface_hash(retgraph.subgraph(interface_graph_nodes))

            core_graph_nodes = [item for x in range(radius_ + 1) for item in nodedict.get(x, [])]
            corehash = calc_interface_hash(retgraph.subgraph(core_graph_nodes))

            #get relevant subgraph
            nodes = [node for i in range(radius_ + thickness_ + 1) for node in nodedict[i]]
            thisgraph = nx.Graph(retgraph.subgraph(nodes))

            #marking cores and interfaces in subgraphs
            for i in range(radius_ + 1):
                for no in nodedict[i]:
                    thisgraph.node[no]['core'] = True
                    if 'interface' in thisgraph.node[no]:
                        thisgraph.node[no].pop('interface')
            for i in range(radius_ + 1, radius_ + thickness_ + 1):
                if i in nodedict:
                    for no in nodedict[i]:
                        thisgraph.node[no]['interface'] = True
                        if 'core' in thisgraph.node[no]:
                            thisgraph.node[no].pop('core')

            retlist.append(core_interface_data(interfacehash, corehash, thisgraph, radius_, thickness_))
    return retlist








    

