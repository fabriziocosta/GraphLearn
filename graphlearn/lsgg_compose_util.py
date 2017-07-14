
from toolz.functoolz import memoize
import eden.graph as eg

# note on the memoize... using id(args) fails,, not sure why


@memoize(key=lambda args, kwargs: args)
def _add_hlabel(graph):
    eg._label_preprocessing(graph)


@memoize(key=lambda args, kwargs: args)
def _edge_to_vertex(graph):
    return eg._edge_to_vertex_transform(graph)


from eden import fast_hash
import networkx as nx


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
                 distance_dict={}):
        self.interface_hash = interface_hash
        self.core_hash = core_hash
        self.graph = graph
        self.radius = radius
        self.thickness = thickness
        self.core_nodes_count = core_nodes_count
        self.count = count  # will be used to count how often we see this during grammar creation
        self.distance_dict = distance_dict  # this attribute is slightly questionable. maybe remove it?

    def __str__(self):
        return 'cip: int:%d, cor:%d, rad:%d, thi:%d, rot:%d' % \
               (self.interface_hash, self.core_hash, self.radius, self.thickness, min(self.distance_dict.get(0, [999])))


################
#  decompose
###############

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


def graph_hash(graph, hash_bitmask, node_name_label=lambda id, node: node['hlabel']):
    """
        so we calculate a hash of a graph
    """
    l = []
    node_name_cache = {}
    all_nodes = set(graph.nodes())
    visited = set()
    # all the edges
    for (a, b) in graph.edges():
        visited.add(a)
        visited.add(b)

        ha = node_name_cache.get(a, -1)
        if ha == -1:
            ha = calc_node_name(graph, a, hash_bitmask, node_name_label)
            node_name_cache[a] = ha
        hb = node_name_cache.get(b, -1)
        if hb == -1:
            hb = calc_node_name(graph, b, hash_bitmask, node_name_label)
            node_name_cache[b] = hb
        l.append((ha ^ hb) + (ha + hb))

    l.sort()

    # nodes that dont have edges
    z = [node_name_label(node_id, graph.node[node_id]) for node_id in all_nodes - visited]
    z.sort()
    ihash = fast_hash(l + z, hash_bitmask)
    return ihash


def calc_node_name(interfacegraph, node, hash_bitmask, node_name_label=lambda id, node: node['hlabel']):
    '''
     part of generating the hash for a graph is calculating therhash of a node in the graph
    '''
    d = nx.single_source_shortest_path_length(interfacegraph, node, 20)
    l = [node_name_label(nid, interfacegraph.node[nid]) + dis for nid, dis in d.items()]
    l.sort()
    l = fast_hash(l, hash_bitmask)
    return l


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
    graph = _edge_to_vertex(graph)
    _add_hlabel(graph)

    # dict of relevant nodes
    horizon = radius + thickness
    dist = nx.single_source_shortest_path_length(graph, root_node, horizon)
    node_dict = invert_dict(dist)

    # core hash
    core_graph_nodes = [item
                        for x in range(radius + 1)
                        for item in node_dict.get(x, [])]
    core_hash = graph_hash(graph.subgraph(core_graph_nodes), hash_bitmask)

    # interface hash
    interface_graph_nodes = [item
                             for x in range(radius + 1, radius + thickness + 1)
                             for item in node_dict.get(x, [])]
    subgraph = graph.subgraph(interface_graph_nodes)
    interface_hash = graph_hash(subgraph,
                                hash_bitmask,
                                node_name_label=lambda id, node: node['hlabel'] + dist[id] - radius)

    # make cip
    nodes = [node for i in range(radius + thickness + 1) for node in node_dict.get(i, [])]
    cip_graph = graph.subgraph(nodes).copy()
    for i in range(radius + 1):
        for no in node_dict.get(i, []):
            cip_graph.node[no]['core'] = True

    for i, di in enumerate(range(radius + 1, radius + thickness + 1)):
        for no in node_dict.get(di, []):
            cip_graph.node[no]['interface'] = True
            cip_graph.node[no]['distance_dependent_label'] = cip_graph.node[no]['hlabel'] + i

    # return a cip thing :)
    core_nodes_count = sum([len(node_dict.get(x, [])) for x in range(radius + 1)])
    return CoreInterfacePair(interface_hash,
                             core_hash,
                             cip_graph,
                             radius,
                             thickness,
                             core_nodes_count,
                             distance_dict=node_dict)

######
# compose
#####import networkx as nx
from networkx.algorithms import isomorphism as iso
import logging
logger = logging.getLogger(__name__)


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

    #graph.node[node]['interface'] = True
    graph.remove_node(node2)


def find_all_isomorphisms(home, other):
    if iso.faster_could_be_isomorphic(home, other):

        label_matcher = lambda x, y: x['distance_dependent_label'] == y['distance_dependent_label'] and \
            x.get('shard', 1) == y.get('shard', 1)

        graph_label_matcher = iso.GraphMatcher(home, other, node_match=label_matcher)
        for index, mapping in enumerate(graph_label_matcher.isomorphisms_iter()):
            if index == 15:  # give up ..
                break
            yield mapping
    else:
        logger.debug('faster iso check failed')
        raise StopIteration


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


def core_substitution(graph, orig_cip_graph, new_cip_graph):
    """
    graph is the whole graph..
    subgraph is the interfaceregrion in that we will transplant
    new_cip_graph which is the interface and the new core
    """
    graph = _edge_to_vertex(graph)
    assert(set(orig_cip_graph.nodes()) - set(graph.nodes()) == set([])), 'orig_cip_graph not in graph'

    # select only the interfaces of the cips
    new_graph_interface_nodes = [n for n, d in new_cip_graph.nodes(data=True) if 'core' not in d]
    new_cip_interface_graph = nx.subgraph(new_cip_graph, new_graph_interface_nodes)

    original_graph_interface_nodes = [n for n, d in orig_cip_graph.nodes(data=True) if 'core' not in d]
    original_interface_graph = nx.subgraph(orig_cip_graph, original_graph_interface_nodes)
    # get isomorphism between interfaces, if none is found we return an empty graph

    iso = get_good_isomorphism(graph,
                               orig_cip_graph,
                               new_cip_graph,
                               original_interface_graph,
                               new_cip_interface_graph)

    if len(iso) != len(original_interface_graph):
        # print iso
        # draw.display(orig_cip_graph)
        # draw.display(new_cip_graph)
        #draw.graphlearn([orig_cip_graph, new_cip_graph],size=10)
        logger.log(5, "grammar hash collision, discovered in 'core_substution' ")
        return None

    # ok we got an isomorphism so lets do the merging
    graph = nx.union(graph, new_cip_graph, rename=('', '-'))

    # removing old core
    # original_graph_core_nodes = [n for n, d in orig_cip_graph.nodes(data=True) if 'core' in d]
    original_graph_core_nodes = [n for n, d in orig_cip_graph.nodes(data=True) if 'core' in d]

    for n in original_graph_core_nodes:
        graph.remove_node(str(n))

    # merge interfaces
    for k, v in iso.iteritems():
        # graph.node[str(k)][
        #    'intgggerface'] = True  # i am marking the interface only for the backflow probability calculation in graphlearn, this is probably deleteable because we also do this in merge, also this line is superlong Ooo
        merge(graph, str(k), '-' + str(v))
    # unionizing killed my labels so we need to relabel

    graph = eg._revert_edge_to_vertex_transform(graph)
    re = nx.convert_node_labels_to_integers(graph)
    graph_clean(re)

    return re
