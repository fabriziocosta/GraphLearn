import networkx as nx
from networkx.algorithms import isomorphism as iso
from eden import fast_hash

from grammar import core_interface_pair

#####################################   extract a core/interface pair #####################


def invert_dict(d):
    """
    so input is usualy a distance dictionaty so
    {nodenumber: distance, nodenumber:distance} we turn this into {distance: [nodenumber, nodenumber]}
    """
    d2 = {}
    for k, v in d.iteritems():
        l = []
        d2[v] = d2.get(v, l)
        d2[v].append(k)
    return d2


def calc_interface_hash(interface_graph, hash_bitmask):
    """
        so we calculate a hash of a graph
    """
    l = []
    node_name_cache = {}

    all_nodes = set(interface_graph.nodes())
    visited = set()
    # all the edges
    for (a, b) in interface_graph.edges():
        visited.add(a)
        visited.add(b)

        ha = node_name_cache.get(a, -1)
        if ha == -1:
            ha = calc_node_name(interface_graph, a, hash_bitmask)
            node_name_cache[a] = ha
        hb = node_name_cache.get(b, -1)
        if hb == -1:
            hb = calc_node_name(interface_graph, b, hash_bitmask)
            node_name_cache[b] = hb
        l.append((ha ^ hb) + (ha + hb))
    l.sort()

    # nodes that dont have edges
    l += [interface_graph.node[node_id]['hlabel'][0] for node_id in all_nodes - visited]
    l = fast_hash(l, hash_bitmask)
    return l


def calc_core_hash(core_graph, hash_bitmask):
    return calc_interface_hash(core_graph, hash_bitmask)


def calc_node_name(interfacegraph, node, hash_bitmask):
    '''
     part of generating the hash for a graph is calculating the hash of a node in the graph
    '''
    d = nx.single_source_shortest_path_length(interfacegraph, node, 20)
    # d is now node:dist
    # l is a list of  hash(label,distance)
    #l=[   func([interfacegraph.node[nid]['intlabel'],dis])  for nid,dis in d.items()]
    l = [interfacegraph.node[nid]['hlabel'][0] + dis for nid, dis in d.items()]
    l.sort()
    l = fast_hash(l, hash_bitmask)
    return l


def extract_core_and_interface(root_node, graph, radius_list=None, thickness_list=None, vectorizer=None,
                               hash_bitmask=2 * 20 - 1):
    """

:param root_node: root root_node
:param graph: graph
:param radius_list:
:param thickness_list:
:param vectorizer: a vectorizer
:param hash_bitmask:


:return: radius_list*thicknes_list long list of cips
"""

    if 'hlabel' not in graph.node[0]:
        vectorizer._label_preprocessing(graph)

    # which nodes are in the relevant radius
    dist = nx.single_source_shortest_path_length(graph, root_node, max(radius_list) + max(thickness_list))
    # we want the relevant subgraph and we want to work on a copy
    master_cip_graph = nx.Graph(graph.subgraph(dist))

    # we want to inverse the dictionary.
    # so now we see {distance:[list of nodes at that distance]}
    nodedict = invert_dict(dist)

    cip_list = []
    for thickness_ in thickness_list:
        for radius_ in radius_list:

            # see if it is feasable to extract
            if radius_ + thickness_ not in nodedict:
                continue

            # calculate hashes
            # d={1:[1,2,3],2:[3,4,5]}
            # print [ i for x in [1,2] for i in d[x] ]
            interface_graph_nodes = [item for x in range(radius_ + 1, radius_ + thickness_ + 1) for item in
                                     nodedict.get(x, [])]
            interfacehash = calc_interface_hash(master_cip_graph.subgraph(interface_graph_nodes), hash_bitmask)

            core_graph_nodes = [item for x in range(radius_ + 1) for item in nodedict.get(x, [])]
            corehash = calc_core_hash(master_cip_graph.subgraph(core_graph_nodes), hash_bitmask)

            # get relevant subgraph
            nodes = [node for i in range(radius_ + thickness_ + 1) for node in nodedict[i]]
            cip_graph = nx.Graph(master_cip_graph.subgraph(nodes))

            # marking cores and interfaces in subgraphs
            for i in range(radius_ + 1):
                for no in nodedict[i]:
                    cip_graph.node[no]['core'] = True
                    if 'interface' in cip_graph.node[no]:
                        cip_graph.node[no].pop('interface')
            for i in range(radius_ + 1, radius_ + thickness_ + 1):
                if i in nodedict:
                    for no in nodedict[i]:
                        cip_graph.node[no]['interface'] = True
                        if 'core' in cip_graph.node[no]:
                            cip_graph.node[no].pop('core')

            core_nodes_count = sum([len(nodedict[x]) for x in range(radius_ + 1)])

            cip_list.append(core_interface_pair(interfacehash, corehash, cip_graph, radius_, thickness_, core_nodes_count, distance_dict=nodedict))
    return cip_list



###########################  core substitution  ####################

def merge(G, node, node2):
    '''
    merge node2 into the node.
    input nodes are strings,
    node is the king
    '''
    for n in G.neighbors(node2):
        G.add_edge(node, n)
    G.node[node]['interface'] = True
    G.remove_node(node2)

def find_isomorphism( home, other):
    matcher = lambda x, y: x['label'] == y['label']
    GM = iso.GraphMatcher(home, other, node_match=matcher)
    if GM.is_isomorphic() == False:
        return {}
    return GM.mapping

def core_substitution( graph, original_cip_graph, new_cip_graph):
    """
    graph is the whole graph..
    subgraph is the interfaceregrion in that we will transplant
    new_cip_graph which is the interface and the new core
    """
    # select only the interfaces of the cips
    nocore = [n for n, d in new_cip_graph.nodes(data=True) if d.has_key('core') == False]
    newgraph_interface = nx.subgraph(new_cip_graph, nocore)
    nocore = [n for n, d in original_cip_graph.nodes(data=True) if d.has_key('core') == False]
    subgraph_interface = nx.subgraph(original_cip_graph, nocore)
    # get isomorphism between interfaces, if none is found we return an empty graph
    iso = find_isomorphism(subgraph_interface, newgraph_interface)
    if len(iso) != len(subgraph_interface):
        return nx.Graph()
    # ok we got an isomorphism so lets do the merging
    G = nx.union(graph, new_cip_graph, rename=('', '-'))
    # removing old core
    nocore = [n for n, d in original_cip_graph.nodes(data=True) if d.has_key('core')]
    for n in nocore:
        G.remove_node(str(n))
    # merge interfaces
    for k, v in iso.iteritems():
        merge(G, str(k), '-' + str(v))
    # unionizing killed my labels so we need to relabel
    return nx.convert_node_labels_to_integers(G)


def graph_clean(graph):
    '''
    in the precess of creating a new graph,
    we marked the nodes that were used as interface and core.
    here we remove the marks.
    :param graph:
    :return:
    '''
    for n, d in graph.nodes(data=True):
        d.pop('core', None)
        d.pop('interface', None)