
from toolz.functoolz import memoize
import eden.graph as eg
from networkx.algorithms import isomorphism as iso
import logging
logger = logging.getLogger(__name__)
from eden import fast_hash
import networkx as nx



@memoize(key=lambda args, kwargs: args)
def _add_hlabel(graph):
    eg._label_preprocessing(graph)


@memoize(key=lambda args, kwargs: args)
def _edge_to_vertex(graph):
    return eg._edge_to_vertex_transform(graph)


class CoreInterfacePair:
    """
    this is referred to throughout the code as cip
    it contains the cip-graph and several pieces of information about it.
    """

    def __init__(self,
                 interface_hash=0,
                 core_hash=0,
                 graph=None,
                 radius=0,
                 thickness=0,
                 core_nodes_count=0,
                 count=0,
                 root=None,
                 core_nodes=[],interface_nodes=[],interface_graph=None):
        self.interface_hash = interface_hash
        self.core_hash = core_hash
        self.graph = graph
        self.radius = radius
        self.thickness = thickness
        self.core_nodes_count = core_nodes_count
        self.count = count  # will be used to count how often we see this during grammar creation
        self.root = root
        self.interface_nodes=interface_nodes
        self.core_nodes=core_nodes
        self.interface_graph=interface_graph # reference to the graph thing:)

    def __str__(self):
        return 'cip: int:%d, cor:%d, rad:%d, thi:%d, rot:%d' % \
               (self.interface_hash, self.core_hash, self.radius, self.thickness, self.root)


################
#  decompose
###############




def graph_hash(graph, hash_bitmask, node_name_label=lambda id, node: node['hlabel']):
    """
        so we calculate a hash of a graph
    """
    node_names = {n: calc_node_name(graph,n,hash_bitmask,node_name_label) for n in graph.nodes()}

    tmp_fast_hash = lambda x,y: (y ^ x) + (y + x)
    l = [ tmp_fast_hash(node_names[a],node_names[b])  for (a,b) in graph.edges() ]
    l.sort()

    isolates= [n for (n,d) in graph.degree_iter() if d==0]
    z = [node_name_label(node_id, graph.node[node_id]) for node_id in  isolates]
    z.sort()

    return fast_hash(l + z, hash_bitmask)


def calc_node_name(interfacegraph, node, hash_bitmask, node_name_label=lambda id, node: node['hlabel']):
    '''
     part of generating the hash for a graph is calculating therhash of a node in the graph

     # the case that n has no neighbors is currently untested...
    '''
    d = nx.single_source_shortest_path_length(interfacegraph, node, 20)
    l = [node_name_label(nid, interfacegraph.node[nid]) + dis for nid, dis in d.items()]
    l.sort()
    return fast_hash(l, hash_bitmask)




def extract_core_and_interface(root_node=None,
                               graph=None,
                               radius=None,
                               thickness=None,
                               hash_bitmask=2 ** 20 - 1):
    '''

    Parameters
    ----------
    root_node
    graph
    radius
    thickness
    hash_bitmask

    Returns
    -------
        makes a cip oO
    '''
    # preprocessing
    graph = _edge_to_vertex(graph)
    _add_hlabel(graph)
    dist = nx.single_source_shortest_path_length(graph, root_node, radius+thickness)

    # find interesting nodes:
    core_nodes = [  id for id,dst in dist.items() if dst <= radius  ]
    interface_nodes = [id for id,dst in dist.items() if radius<dst<=radius+thickness]
    
    
    # calculate hashes
    core_hash = graph_hash(graph.subgraph(core_nodes), hash_bitmask)
    interface_hash = graph_hash(graph.subgraph(interface_nodes),
                                hash_bitmask,
                                node_name_label=lambda id, node: node['hlabel'] + dist[id] - radius)

    # copy cip and mark core/interface
    cip_graph = graph.subgraph(core_nodes+interface_nodes).copy()
    for no in interface_nodes:
        cip_graph.node[no]['distance_dependent_label'] = cip_graph.node[no]['hlabel'] + dist[no]-(radius+1)

    return CoreInterfacePair(interface_hash,
                             core_hash,
                             cip_graph,
                             radius,
                             thickness,
                             len(core_nodes),
                             root=root_node,
                             core_nodes=core_nodes,
                             interface_nodes=interface_nodes,
                             interface_graph=nx.subgraph(cip_graph,interface_nodes))



######
# compose
######



def merge(graph, node, node2):
    '''
    merge node2 into the node.
    input nodes are strings,
    node is the king
    '''

    for n in graph.neighbors(node2):
        graph.add_edge(node, n)

    if isinstance(graph, nx.DiGraph):
        for n in graph.predecessors(node2):
            graph.add_edge(n, node)

    graph.remove_node(node2)


def find_all_isomorphisms(home, other):
    if iso.faster_could_be_isomorphic(home, other):

        label_matcher = lambda x, y: x['distance_dependent_label'] == y['distance_dependent_label'] and \
            x.get('shard', 1) == y.get('shard', 1)

        graph_label_matcher = iso.GraphMatcher(home, other, node_match=label_matcher)
        for index, mapping in enumerate(graph_label_matcher.isomorphisms_iter()):
            if index == 5:
                logger.debug('lsgg_compose_util i checked more than 5 isomorphisms')
            yield mapping
    else:
        logger.debug('lsgg_compose_util faster iso check failed')
        raise StopIteration


def core_substitution(graph,orig_cip, new_cip):
    """
    graph is the whole graph..
    subgraph is the interfaceregrion in that we will transplant
    new_cip_graph which is the interface and the new core
    """
    # preprocess
    graph = _edge_to_vertex(graph)
    assert(set(orig_cip.graph.nodes()) - set(graph.nodes()) == set([])), 'lsgg_compose_util orig_cip_graph not in graph'


    # get isomorphism
    iso = find_all_isomorphisms(orig_cip.interface_graph, new_cip.interface_graph).next()
    if len(iso) != len(orig_cip.interface_graph):
        logger.log(5, "lsgg_compose_util grammar hash collision, discovered in 'core_substution' ")
        return None


    # make graph union (the old graph and the new cip are now floating side by side)
    graph = nx.union(graph, new_cip.graph, rename=('', '-'))
    graph.remove_nodes_from( map(str,orig_cip.core_nodes))
    # merge interface nodes
    for k, v in iso.iteritems():
        merge(graph, str(k), '-' + str(v))

    graph = eg._revert_edge_to_vertex_transform(graph)
    re = nx.convert_node_labels_to_integers(graph)
    return re

"""
def get_good_isomorphism(graph, orig_cip_graph, new_cip_graph, home, other):
    '''
    we need isomorphisms between two interfaces, networkx is able to calculate these.
    we use these isomorphism mappings to do the core-replacement.
    some mappings will cause the core replacement to violate the 'edge-nodes have exactly 2 neighbors'
    constraint.

    here we filter those out.
    :param graph: a graph
    :cip-cip whatever: obvious
    :param home: the interface in the home graph
    :param other: the interface of a new cip
    :return: a dictionary that is either empty or a good isomorphism

    update 23.7.15: not sure if this is a problem anymore//
    undate 29.07.15: with thickness .5 things go wrong when directed because the interfacenode
    just has no direction indicator
    update 2017-07-15: this seems to do some special checking for
    digraphs. today i rewrote the cip replacement code, and dont have a testcase for digraphs
    so i deactivate this for now
    '''

    if isinstance(home, nx.DiGraph):
        # for mapping in find_all_isomorphisms(home, other):
        #    return mapping

        # for all the mappings home-> other
        for i, mapping in enumerate(find_all_isomorphisms(home, other)):
            for home_node in mapping.keys():
                # check if all the edge nodes are not violating anything
                if 'edge' in graph.node[home_node]:

                    # neighbors onthe outside
                    old_neigh = len([e for e in graph.neighbors(home_node) if e not in orig_cip_graph.node])
                    # neighbors on the inside
                    new_neigh = len([e for e in new_cip_graph.neighbors(mapping[home_node])])

                    # predec onthe outside
                    old_pre = len([e for e in graph.predecessors(home_node) if e not in orig_cip_graph.node])
                    # predec on the inside
                    new_pre = len([e for e in new_cip_graph.predecessors(mapping[home_node])])

                    # an edge node should have at least one outging and one incoming edge...
                    if old_neigh + new_neigh == 0 or old_pre + new_pre == 0:
                        break
            else:
                if i > 0:
                    logger.log(5, 'isomorphism #%d accepted' % i)

                return mapping

    else:
        for mapping in find_all_isomorphisms(home, other):
            return mapping
    return {}
"""
