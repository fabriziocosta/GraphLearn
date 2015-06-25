import networkx as nx
from networkx.algorithms import isomorphism as iso
from eden import fast_hash
from coreinterfacepair import CoreInterfacePair
import logging
#import graphlearn.utils.draw as myutils

logger = logging.getLogger(__name__)


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
        # z=(ha ^ hb) + (ha + hb)
        # l.append( fast_hash([ha,hb],hash_bitmask) +z )
    l.sort()

    # nodes that dont have edges
    z = [interface_graph.node[node_id]['hlabel'][0] for node_id in all_nodes - visited]
    z.sort()
    ihash = fast_hash(l + z, hash_bitmask)
    return ihash


def calc_core_hash(core_graph, hash_bitmask):
    return calc_interface_hash(core_graph, hash_bitmask)


def calc_node_name(interfacegraph, node, hash_bitmask):
    '''
     part of generating the hash for a graph is calculating the hash of a node in the graph
    '''
    d = nx.single_source_shortest_path_length(interfacegraph, node, 20)
    # d is now node:dist
    # l is a list of  hash(label,distance)
    # l=[   func([interfacegraph.node[nid]['intlabel'],dis])  for nid,dis in d.items()]
    l = [interfacegraph.node[nid]['hlabel'][0] + dis for nid, dis in d.items()]
    l.sort()
    l = fast_hash(l, hash_bitmask)
    return l


def extract_core_and_interface(root_node, graph, radius_list=None, thickness_list=None, vectorizer=None,
                               hash_bitmask=2 ** 20 - 1, filter=lambda x, y: True):
    """

:param root_node: root root_node
:param graph: graph
:param radius_list:
:param thickness_list:
:param vectorizer: a vectorizer
:param hash_bitmask:


:return: radius_list*thicknes_list long list of cips
"""

    if not filter(graph, root_node):
        return []
    if 'hlabel' not in graph.node[ graph.nodes()[0] ]:
        vectorizer._label_preprocessing(graph)

    # which nodes are in the relevant radius
    #print root_node,max(radius_list) + max(thickness_list)
    #myutils.display(graph,vertex_label='id',size=15)


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

            core_graph_nodes = [item for x in range(radius_ + 1) for item in nodedict.get(x, [])]
            if not filter(master_cip_graph, core_graph_nodes):
                continue

            corehash = calc_core_hash(master_cip_graph.subgraph(core_graph_nodes), hash_bitmask)

            interface_graph_nodes = [item for x in range(radius_ + 1, radius_ + thickness_ + 1) for item in
                                     nodedict.get(x, [])]
            interfacehash = calc_interface_hash(master_cip_graph.subgraph(interface_graph_nodes), hash_bitmask)

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

            cip_list.append(CoreInterfacePair(interfacehash, corehash, cip_graph, radius_, thickness_, core_nodes_count,
                                              distance_dict=nodedict))
    return cip_list


def filter(graph, nodes):
    # we say true if the graph is ok
    # root node?
    if type(nodes) != list:
        if 'no_root' in graph.node[nodes]:
            return False
        else:
            return True
    else:
        for node in nodes:
            if 'not_in_core' in graph.node[node]:
                return False
        return True


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


def find_isomorphism(home, other):
    matcher = lambda x, y: x['label'] == y['label']
    GM = iso.GraphMatcher(home, other, node_match=matcher)
    if GM.is_isomorphic() is False:
        return {}
    return GM.mapping


def core_substitution(graph, original_cip_graph, new_cip_graph):
    """
    graph is the whole graph..
    subgraph is the interfaceregrion in that we will transplant
    new_cip_graph which is the interface and the new core
    """
    # select only the interfaces of the cips
    new_graph_interface_nodes = [n for n, d in new_cip_graph.nodes(data=True) if 'core' not in d]
    new_cip_interface_graph = nx.subgraph(new_cip_graph, new_graph_interface_nodes)

    original_graph_interface_nodes = [n for n, d in original_cip_graph.nodes(data=True) if 'core' not in d]
    original_interface_graph = nx.subgraph(original_cip_graph, original_graph_interface_nodes)
    # get isomorphism between interfaces, if none is found we return an empty graph
    iso = find_isomorphism(original_interface_graph, new_cip_interface_graph)
    if len(iso) != len(original_interface_graph):
        # drawgraphs([graph,original_cip_graph,new_cip_graph],contract=False)
        return nx.Graph()
    # ok we got an isomorphism so lets do the merging
    G = nx.union(graph, new_cip_graph, rename=('', '-'))
    # removing old core
    original_graph_core_nodes = [n for n, d in original_cip_graph.nodes(data=True) if 'core' in d]
    for n in original_graph_core_nodes:
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
        d.pop('root', None)
