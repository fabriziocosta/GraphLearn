import networkx as nx
from networkx.algorithms import isomorphism as iso
from eden import fast_hash
from coreinterfacepair import CoreInterfacePair
import logging
import traceback
import utils.draw as draw
from eden.graph import Vectorizer

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


def calc_interface_hash(interface_graph, hash_bitmask,node_name_label=None):
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
            ha = calc_node_name(interface_graph, a, hash_bitmask, node_name_label)
            node_name_cache[a] = ha
        hb = node_name_cache.get(b, -1)
        if hb == -1:
            hb = calc_node_name(interface_graph, b, hash_bitmask, node_name_label)
            node_name_cache[b] = hb
        l.append((ha ^ hb) + (ha + hb))
        # z=(ha ^ hb) + (ha + hb)
        # l.append( fast_hash([ha,hb],hash_bitmask) +z )
    l.sort()

    # nodes that dont have edges
    if node_name_label==None:
        z = [interface_graph.node[node_id]['hlabel'][0] for node_id in all_nodes - visited]
    else:
        z = [interface_graph.node[node_id][node_name_label] for node_id in all_nodes - visited]
    z.sort()
    ihash = fast_hash(l + z, hash_bitmask)
    return ihash

def calc_node_name(interfacegraph, node, hash_bitmask,node_name_label):
    '''
     part of generating the hash for a graph is calculating the hash of a node in the graph
    '''
    d = nx.single_source_shortest_path_length(interfacegraph, node, 20)
    # d is now node:dist
    # l is a list of  hash(label,distance)
    # l=[   func([interfacegraph.node[nid]['intlabel'],dis])  for nid,dis in d.items()]
    if node_name_label == None:
        l = [interfacegraph.node[nid]['hlabel'][0] + dis  for nid, dis in d.items()]
    else:
        l = [interfacegraph.node[nid][node_name_label] + dis  for nid, dis in d.items()]
    l.sort()
    l = fast_hash(l, hash_bitmask)
    return l


def calc_core_hash(core_graph, hash_bitmask,**kwargs):
    return calc_interface_hash(core_graph, hash_bitmask,**kwargs)




def extract_core_and_interface(root_node=None, graph=None, radius_list=None, thickness_list=None, vectorizer=Vectorizer(),
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



    undir_graph = nx.Graph(graph)
    dist = nx.single_source_shortest_path_length(undir_graph, root_node, max(radius_list) + max(thickness_list))
    # we want the relevant subgraph and we want to work on a copy
    master_cip_graph = graph.subgraph(dist).copy()

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

            for inode in interface_graph_nodes:
                master_cip_graph.node[inode]['temporary_substitution_label'] = master_cip_graph.node[inode]['hlabel'][0] + dist[inode] - radius_

            interfacehash = calc_interface_hash(master_cip_graph.subgraph(interface_graph_nodes), hash_bitmask, node_name_label='temporary_substitution_label')

            # get relevant subgraph
            nodes = [node for i in range(radius_ + thickness_ + 1) for node in nodedict[i]]
            cip_graph = master_cip_graph.subgraph(nodes).copy()

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
    if isinstance(G,nx.DiGraph):
        for n in G.predecessors(node2):
            G.add_edge(n,node)

    G.node[node]['interface'] = True
    G.remove_node(node2)


def find_all_isomorphisms(home, other):

    if iso.faster_could_be_isomorphic(home,other):
        matcher = lambda x, y: x['temporary_substitution_label'] == y['temporary_substitution_label']
        GM = iso.GraphMatcher(home, other, node_match=matcher)
        for index,mapping in enumerate(GM.isomorphisms_iter()):
            if index > 1:
                logger.debug('delivering isomorphism nr.%s' % index )
            #if index==10: # give up ..
            #    break
            yield mapping
    else:
        logger.debug('faster iso check failed')



def get_good_isomorphism(graph,original_cip_graph,new_cip_graph,home,other):
    '''
    we need isomorphisms between two interfaces, netowrkx is able to calculate these.
    we use these isomorphism mappings to do the core-replacement.
    some mappings will cause the core replacement to violate the  'edge-nodes have exactly 2 neighbors' constraint.
    here we filter those out.
    :param graph: a graph
    :cip-cip whatever: obvious
    :param home: the interface in the home graph
    :param other: the interface of a new cip
    :return: a dictionary that is either empty or a good isomorphism
    '''
    if isinstance(home, nx.DiGraph):
        #for mapping in find_all_isomorphisms(home,other):
        #    return mapping


        # this is probably broken  ASDASD
        for mapping in find_all_isomorphisms(home,other):
            for home_node in mapping.keys():
                if 'edge' in graph.node[home_node]:
                    old_neigh = len([e for e in  graph.neighbors(home_node) if e not in original_cip_graph.node ])
                    new_neigh = len([e for e in  new_cip_graph.neighbors(mapping[home_node])])
                    if 0 < (old_neigh+new_neigh) <3 : # we have a directed graph so 1 and 2 neighbors are ok
                        break
            else:
                return mapping
    else:
        for mapping in find_all_isomorphisms(home,other):
            for home_node in mapping.keys():
                if 'edge' in graph.node[home_node]:
                    old_neigh = len([e for e in  graph.neighbors(home_node) if e not in original_cip_graph.node ])
                    new_neigh = len([e for e in  new_cip_graph.neighbors(mapping[home_node])])
                    if old_neigh+new_neigh != 2:
                        break
            # we didnt break so every edge is save
            else:
                return mapping
        # draw rejected pair:
        #draw.draw_graph_set_graphlearn([original_cip_graph,new_cip_graph])
    return {}


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

    iso = get_good_isomorphism(graph,original_cip_graph,new_cip_graph,original_interface_graph, new_cip_interface_graph)

    if len(iso) != len(original_interface_graph):
        #print iso
        #draw.display(original_cip_graph)
        #draw.display(new_cip_graph)
        return nx.Graph()

    # ok we got an isomorphism so lets do the merging
    G = nx.union(graph, new_cip_graph, rename=('', '-'))

    # removing old core
    #original_graph_core_nodes = [n for n, d in original_cip_graph.nodes(data=True) if 'core' in d]
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




'''
  this is just a test to see if we can use an the estimator stuff to calculate the interface hash.
    the experiment failed.
'''
def extract_core_and_interface2(root_node, graph, radius_list=None, thickness_list=None, vectorizer=None,
                               hash_bitmask=2 ** 20 - 1, filter=lambda x, y: True,esti=None):
    """


    this is just a test to see if we can use an the estimator stuff to calculate the interface hash.
    the experiment failed.

    :param root_node: root root_node
    :param graph: graph
    :param radius_list:
    :param thickness_list:
    :param vectorizer: a vectorizer
    :param hash_bitmask:
    :return: radius_list*thicknes_list long list of cips
    """
    try:
        if not filter(graph, root_node):
            return []
        if 'hlabel' not in graph.node[ graph.nodes()[0] ]:
            vectorizer._label_preprocessing(graph)

        # which nodes are in the relevant radius
        #print root_node,max(radius_list) + max(thickness_list)
        #myutils.display(graph,vertex_label='id',size=15)


        dist = nx.single_source_shortest_path_length(graph, root_node, max(radius_list) + max(thickness_list))
        # we want the relevant subgraph and we want to work on a copy
        master_cip_graph = graph.subgraph(dist).copy()

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


                #interfacehash = calc_interface_hash(master_cip_graph.subgraph(interface_graph_nodes), hash_bitmask)
                interfacehash= round(esti.predict_proba(vectorizer.transform_single(master_cip_graph.subgraph(interface_graph_nodes).copy()))[0,0],7)


                # get relevant subgraph
                nodes = [node for i in range(radius_ + thickness_ + 1) for node in nodedict[i]]
                cip_graph = master_cip_graph.subgraph(nodes).copy()

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

    except Exception as exc:
            print traceback.format_exc(10)