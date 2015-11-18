import networkx as nx
from networkx.algorithms import isomorphism as iso
from eden import fast_hash
from coreinterfacepair import CoreInterfacePair
import logging
from eden.graph import Vectorizer
import random

logger = logging.getLogger(__name__)




class AbstractWrapper(object):

    def rooted_core_interface_pairs(self, root, **args):
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

    def graph(self):
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
        logger.log(__debug__,'you may want to implement mark_median')

    def out(self):
        '''
        :return: the sampling process of graphlearn outputs a result.
                here we create this result. this result may be anything
                eg graph, string
        '''
        raise NotImplementedError("Should have implemented this")



    def random_core_interface_pair(self,radius_list=None,thickness_list=None, **args):
        '''
        :param radius_list:
        :param thickness_list:
        :param args: args for the extraction ...
        :return: will atempt only once, returns a random cip from our graph or []
        '''
        raise NotImplementedError("Should have implemented this")

    def all_core_interface_pairs(self,**args):
        raise NotImplementedError("Should have implemented this")






class Wrapper(AbstractWrapper):

    def __str__(self):
        return "base_graph size: %s" % len(self._base_graph)

    def __init__(self,graph,vectorizer):
        self.vectorizer=vectorizer
        self._base_graph=graph

    def base_graph(self):
        return self._base_graph

    def rooted_core_interface_pairs(self, root, **args):
        return extract_core_and_interface(root,self._base_graph,vectorizer=self.vectorizer,**args)

    def core_substitution(self, orig_cip_graph, new_cip_graph):
        graph=core_substitution( self._base_graph, orig_cip_graph ,new_cip_graph )
        return graph#self.__class__( graph, self.vectorizer,other=self)


    def mark_median(self, inp='importance', out='is_good', estimator=None):

        graph2 = self._base_graph.copy()  # annotate kills the graph i assume
        graph2 = self.vectorizer.annotate([graph2], estimator=estimator).next()

        for n, d in graph2.nodes(data=True):
            if 'edge' not in d:
                self._base_graph.node[n][inp] = d['importance']

        mark_median(self._base_graph, inp=inp, out=out)


    def clean(self):
        graph_clean(self._base_graph)

    def graph(self):
        return self._base_graph

    def out(self):
        # copy and  if digraph make graph
        graph=nx.Graph(self._base_graph)
        graph= self.vectorizer._revert_edge_to_vertex_transform(graph)
        graph.graph['score']=self.__dict__.get("_score","?")
        return graph

    def random_core_interface_pair(self,radius_list=None,thickness_list=None, **args):

        node = random.choice(self._base_graph.nodes())
        if 'edge' in self._base_graph.node[node]:
            node = random.choice(self._base_graph.neighbors(node))
            # random radius and thickness
        args['radius_list'] = [random.choice(radius_list)]
        args['thickness_list'] = [random.choice(thickness_list)]

        return self.rooted_core_interface_pairs(node, **args)


    def all_core_interface_pairs(self,**args):

        graph=self._base_graph
        cips = []
        for root_node in graph.nodes_iter():
            if 'edge' in graph.node[root_node]:
                continue
            cip_list = self.rooted_core_interface_pairs(root_node,**args)
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
        l.append((ha ^ hb) + (ha + hb))
        # z=(ha ^ hb) + (ha + hb)
        # l.append( fast_hash([ha,hb],hash_bitmask) +z )
    l.sort()

    # nodes that dont have edges
    if node_name_label is None:
        z = [graph.node[node_id]['hlabel'][-1] for node_id in all_nodes - visited]
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
        l = [interfacegraph.node[nid]['hlabel'][-1] + dis for nid, dis in d.items()]
    else:
        l = [interfacegraph.node[nid][node_name_label] + dis for nid, dis in d.items()]
    l.sort()
    l = fast_hash(l, hash_bitmask)
    return l


def extract_core_and_interface(root_node=None,
                               graph=None,
                               radius_list=None,
                               thickness_list=None,
                               vectorizer=Vectorizer(),
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
    if 'hlabel' not in graph.node[graph.nodes()[-1]]:
        vectorizer._label_preprocessing(graph)

    # which nodes are in the relevant radius
    # print root_node,max(radius_list) + max(thickness_list)
    # myutils.display(graph,vertex_label='id',size=15)

    undir_graph = nx.Graph(graph)
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
                label = master_cip_graph.node[inode]['hlabel'][-1]
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
                                     x.get('shard',1)==y.get('shard',1)

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
        graph.node[str(k)]['interface']=True  # i am marking the interface only for the backflow probability calculation in graphlearn, this is probably deleteable because we also do this in merge, also this line is superlong Ooo
        merge(graph, str(k), '-' + str(v))
    # unionizing killed my labels so we need to relabel
    return nx.convert_node_labels_to_integers(graph)


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
