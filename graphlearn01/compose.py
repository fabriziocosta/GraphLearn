import networkx as nx
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

    graph.node[node]['interface'] = True
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
        # i think we cant break here anymore..
        for mapping in find_all_isomorphisms(home, other):
            return mapping
        '''
        for mapping in find_all_isomorphisms(home, other):
            for home_node in mapping.keys():
                if 'edge' in graph.node[home_node]:
                    old_neigh = len([e for e in graph.neighbors(home_node)
                                     if e not in orig_cip_graph.node])
                    new_neigh = len([e for e in new_cip_graph.neighbors(mapping[home_node])])
                    if old_neigh + new_neigh != 2:
                        break
            # we didnt break so every edge is save
            else:
                return mapping
        # draw rejected pair:
        # draw.draw_graph_set_graphlearn([orig_cip_graph,new_cip_graph])
        '''
    return {}


def core_substitution(graph, orig_cip_graph, new_cip_graph):
    """
    graph is the whole graph..
    subgraph is the interfaceregrion in that we will transplant
    new_cip_graph which is the interface and the new core
    """
    assert( set(orig_cip_graph.nodes()) - set(graph.nodes()) == set([]) ), 'orig_cip_graph not in graph'

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
        logger.log(5,"grammar hash collision, discovered in 'core_substution' ")
        return nx.Graph()

    # ok we got an isomorphism so lets do the merging
    graph = nx.union(graph, new_cip_graph, rename=('', '-'))

    # removing old core
    # original_graph_core_nodes = [n for n, d in orig_cip_graph.nodes(data=True) if 'core' in d]
    original_graph_core_nodes = [n for n, d in orig_cip_graph.nodes(data=True) if 'core' in d]

    for n in original_graph_core_nodes:
        graph.remove_node(str(n))

    # merge interfaces
    for k, v in iso.iteritems():
        graph.node[str(k)][
            'interface'] = True  # i am marking the interface only for the backflow probability calculation in graphlearn, this is probably deleteable because we also do this in merge, also this line is superlong Ooo
        merge(graph, str(k), '-' + str(v))
    # unionizing killed my labels so we need to relabel


    return nx.convert_node_labels_to_integers(graph)