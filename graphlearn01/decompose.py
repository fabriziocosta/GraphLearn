'''
provides cip related operations for a graph.
a cip is a part of a graph, cips can be EXTRACTED or REPLACED.


EXTRACTION:
According to EDeN definitions for graph operations a 'decomposer' is a function that takes a graph
and produces many.

REPLACEMENT:
At the end of this document a 'composing' function is found. a composer uses many graphs in input
and returns a single graph.
'''

import random

import networkx as nx
from eden import fast_hash
from eden.graph import Vectorizer
from eden.graph import _label_preprocessing
from eden.graph import _revert_edge_to_vertex_transform

from core_interface_pair import CoreInterfacePair
import compose as Compose
import logging
logger = logging.getLogger(__name__)


class AbstractDecomposer(object):
    def rooted_core_interface_pairs(self, root, radius_list=None,
                                           thickness_list=None):
        '''
        :param root: root node of the cips we want to have
        :param args: specifies radius, thickness and stuff, depends a on implementation
        :return: a list of cips
        '''
        raise NotImplementedError("Should have implemented this")

    def core_substitution(self, orig_cip_graph, new_cip_graph):
        '''
        :param orig_cip_graph: cip (see extract_core_and_interface)
        :param new_cip_graph: cip  ( probably from graphlearn.lsgg )
        :return: graph?  (currently its a graph but graphmanager may be better)
        '''
        raise NotImplementedError("Should have implemented this")

    def base_graph(self):
        '''
        :return: the graph that we want to manipulate
        '''
        raise NotImplementedError("Should have implemented this")

    def clean(self):
        '''
        removes marks on the graph that were made during the core_substitution process.
        this is not done after the substitution because it may be interesting to identify the interface
        that was used :)
        :return: None
        '''
        raise NotImplementedError("Should have implemented this")

    def pre_vectorizer_graph(self):
        '''
        :return: the graph that will be vectorized to work with the estimator
        '''
        raise NotImplementedError("Should have implemented this")

    def mark_median(self, inp='importance', out='is_good'):
        '''
        for each node we look at
        :param inp:
        and decide if
        :param out:
        is marked with 0 or 1 such that half or less of all nodes are marked 1

        :return: Nothing
        '''
        logger.log(__debug__, 'you may want to implement mark_median')

    def out(self):
        '''
        :return: the sampling process of graphlearn outputs a result.
                here we create this result. this result may be anything
                eg graph, string
        '''
        raise NotImplementedError("Should have implemented this")

    def random_core_interface_pair(self,radius_list=None, thickness_list=None,
                                           hash_bitmask=2 ** 20 - 1,
                                           node_filter=lambda x, y: True):
        '''
        :param radius_list:
        :param thickness_list:
        :param args: args for the extraction ...
        :return: will atempt only once, returns a random cip from our graph or []
        '''
        raise NotImplementedError("Should have implemented this")

    def all_core_interface_pairs(self, radius_list=None,
                                           thickness_list=None,
                                           hash_bitmask=2 ** 20 - 1,
                                           node_filter=lambda x, y: True):
        raise NotImplementedError("Should have implemented this")

    def change_basegraph(self,transformerdata):
        raise NotImplementedError("Should have implemented this")



class Decomposer(AbstractDecomposer):
    def __str__(self):
        if "_base_graph" in self.__dict__:
            answer="base_graph with size: %s" % len(self._base_graph)
        else:
            answer='no graphs in decomposer'
        return answer

    def __init__(self, data=[], node_entity_check=lambda x, y: True, nbit=20):
        self._base_graph = data
        self.node_entity_check=node_entity_check
        self.hash_bitmask= 2 ** nbit - 1
        self.nbit=nbit

    def make_new_decomposer(self, transformout):
        return Decomposer(transformout, node_entity_check=self.node_entity_check, nbit=self.nbit)

    def change_basegraph(self,transformerdata):
        self._base_graph=transformerdata


    def base_graph(self):
        return self._base_graph

    def rooted_core_interface_pairs(self, root, radius_list=None,
                                           thickness_list=None):

        return extract_core_and_interface(root_node=root, graph=self._base_graph, radius_list=radius_list,
                                          thickness_list=thickness_list, hash_bitmask=self.hash_bitmask,
                                          node_filter=self.node_entity_check)

    def core_substitution(self, orig_cip_graph, new_cip_graph):
        '''

        Parameters
        ----------
        orig_cip_graph: nx.graph that is a subgraph of the base_graph
        new_cip_graph: nx.graph that is congruent (interface hash matches) to orig_cip_graph

        Returns
        -------
            nx.Graph or nx.DiGraph
            a graph with the new core.
        '''
        graph = Compose.core_substitution(self._base_graph, orig_cip_graph, new_cip_graph)
        return graph

    def mark_median(self, out='is_good', estimator=None,vectorizer=Vectorizer()):

        if type(self._base_graph)==nx.DiGraph:
            graph2 = nx.Graph(self._base_graph)  # annotate kills the graph i assume
        else:
            graph2 = self._base_graph.copy()


        graph2 = vectorizer.annotate([graph2], estimator=estimator).next()

        for n, d in graph2.nodes(data=True):
            if 'edge' not in d:
                self._base_graph.node[n]['markmed_imp'] = d['importance']

        mark_median(self._base_graph, inp='markmed_imp', out=out)

    def clean(self):
        graph_clean(self._base_graph)

    def pre_vectorizer_graph(self):
        # for some reason that i cant see, eden will alter the graph...
        # -> return a copy
        return self._base_graph.copy()

    def out(self):
        # copy and  if digraph make graph
        graph = nx.Graph(self._base_graph)
        graph = _revert_edge_to_vertex_transform(graph)
        graph.graph['score'] = self.__dict__.get("_score", "?")
        return graph

    def random_core_interface_pair(self, radius_list=None, thickness_list=None):


        node = random.choice(filter( lambda x:self.node_entity_check(self._base_graph,x), self._base_graph.nodes()))
        if 'edge' in self._base_graph.node[node]:
            node = random.choice(self._base_graph.neighbors(node))
            # random radius and thickness
        radius_list = [random.choice(radius_list)]
        thickness_list = [random.choice(thickness_list)]

        return self.rooted_core_interface_pairs(node, radius_list=radius_list, thickness_list=thickness_list)

    def all_core_interface_pairs(self,     radius_list=None,
                                           thickness_list=None):

        graph = self._base_graph
        cips = []
        for root_node in graph.nodes_iter():
            if 'edge' in graph.node[root_node]:
                continue
            cip_list = self.rooted_core_interface_pairs(root_node, radius_list=radius_list, thickness_list=thickness_list)
            if cip_list:
                cips.append(cip_list)
        return cips


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


def graph_hash(graph, hash_bitmask, node_name_label=None):
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
        #l.append((ha ^ hb) + (ha + hb))
        blub= lambda ha,hb: (ha ^ hb) + (ha + hb)
        l.append(fast_hash([min(ha,hb),max(ha,hb),blub(ha,hb)],hash_bitmask))

        # z=(ha ^ hb) + (ha + hb)
        # l.append( fast_hash([ha,hb],hash_bitmask) +z )
    l.sort()

    # nodes that dont have edges
    if node_name_label is None:
        z = [graph.node[node_id]['hlabel'] for node_id in all_nodes - visited]
    else:
        z = [graph.node[node_id][node_name_label] for node_id in all_nodes - visited]
    z.sort()
    ihash = fast_hash(l + z, hash_bitmask)
    return ihash


def calc_node_name(interfacegraph, node, hash_bitmask, node_name_label):
    '''
     part of generating the hash for a graph is calculating the hash of a node in the graph
    '''
    d = nx.single_source_shortest_path_length(interfacegraph, node, 20)
    # d is now node:dist
    # l is a list of  hash(label,distance)
    # l=[   func([interfacegraph.node[nid]['intlabel'],dis])  for nid,dis in d.items()]
    if node_name_label is None:
        l = [interfacegraph.node[nid]['hlabel'] + dis for nid, dis in d.items()]
    else:
        l = [interfacegraph.node[nid][node_name_label] + dis for nid, dis in d.items()]
    l.sort()
    l = fast_hash(l, hash_bitmask)
    return l


def extract_core_and_interface(root_node=None, graph=None,
                               radius_list=None,
                               thickness_list=None,
                               hash_bitmask=2 ** 20 - 1,
                               node_filter=lambda x, y: True):
    """
    :param root_node: root root_node
    :param graph: graph
    :param radius_list:
    :param thickness_list:
    :param vectorizer: a vectorizer
    :param hash_bitmask:
    :return: radius_list*thicknes_list long list of cips
    """
    DEBUG = False
    if not node_filter(graph, root_node):
        if DEBUG:
            print 'filta'
        return []
    #print "HELLo!"
    #for n, d in graph.nodes(data=True):
    #    print d
    #if 'hlabel' not in graph.node[graph.nodes()[-1]]:
    _label_preprocessing(graph)
    #for n,d in graph.nodes(data=True):
    #    print d
    # which nodes are in the relevant radius
    # print root_node,max(radius_list) + max(thickness_list)
    # myutils.display(graph,vertex_label='id',size=15)


    if type(graph) == nx.DiGraph:
        undir_graph = nx.Graph(graph)
    else:
        undir_graph = graph

    horizon = max(radius_list) + max(thickness_list)
    dist = nx.single_source_shortest_path_length(undir_graph, root_node, horizon)

    # we want the relevant subgraph and we want to work on a copy
    master_cip_graph = graph.subgraph(dist).copy()

    # we want to inverse the dictionary.
    # so now we see {distance:[list of nodes at that distance]}
    node_dict = invert_dict(dist)

    cip_list = []
    for thickness_ in thickness_list:
        for radius_ in radius_list:
            if DEBUG:
                print 'thickrad', thickness_, radius_
            # see if it is feasable to extract
            if radius_ + thickness_ not in node_dict:
                if DEBUG:
                    print 'jump1', node_dict
                continue

            core_graph_nodes = [item for x in range(radius_ + 1) for item in node_dict.get(x, [])]
            if not node_filter(master_cip_graph, core_graph_nodes):
                if DEBUG:
                    print 'jump2'
                continue

            core_hash = graph_hash(master_cip_graph.subgraph(core_graph_nodes), hash_bitmask)

            interface_graph_nodes = [item for x in range(radius_ + 1, radius_ + thickness_ + 1)
                                     for item in node_dict.get(x, [])]
            for inode in interface_graph_nodes:
                label = master_cip_graph.node[inode]['hlabel']
                master_cip_graph.node[inode]['distance_dependent_label'] = label + dist[inode] - radius_
            subgraph = master_cip_graph.subgraph(interface_graph_nodes)
            interface_hash = graph_hash(subgraph,
                                        hash_bitmask,
                                        node_name_label='distance_dependent_label')

            # get relevant subgraph
            nodes = [node for i in range(radius_ + thickness_ + 1) for node in node_dict[i]]



            cip_graph = master_cip_graph.subgraph(nodes).copy()


            # marking cores and interfaces in subgraphs
            for i in range(radius_ + 1):
                for no in node_dict[i]:
                    cip_graph.node[no]['core'] = True
                    if 'interface' in cip_graph.node[no]:
                        cip_graph.node[no].pop('interface')
            for i in range(radius_ + 1, radius_ + thickness_ + 1):
                if i in node_dict:
                    for no in node_dict[i]:
                        cip_graph.node[no]['interface'] = True
                        if 'core' in cip_graph.node[no]:
                            cip_graph.node[no].pop('core')

            core_nodes_count = sum([len(node_dict[x]) for x in range(radius_ + 1)])

            cip_list.append(CoreInterfacePair(interface_hash,
                                              core_hash,
                                              cip_graph,
                                              radius_,
                                              thickness_,
                                              core_nodes_count,
                                              distance_dict=node_dict))
    return cip_list

''' probably unused .. also the name is problematic, overwriting filter which is used in this file
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
'''

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


def mark_median(graph, inp='importance', out='is_good'):
    # get median
    values = []
    for n, d in graph.nodes(data=True):
        if 'edge' not in d:
            values.append(d[inp])

    # determine cutoff
    values.sort()
    values.append(9999)
    index = len(values) / 2 - 1
    while values[index + 1] == values[index]:
        index += 1
    cutoff = values[index]

    for n, d in graph.nodes(data=True):
        if 'edge' not in d:
            if d[inp] <= cutoff:
                d[out] = 0
            else:
                d[out] = 1
