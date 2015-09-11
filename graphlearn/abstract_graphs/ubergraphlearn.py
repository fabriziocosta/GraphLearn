from eden.modifier.graph import vertex_attributes
from eden.modifier.graph.structure import contraction
import graphlearn.graphtools as graphtools
import random
from graphlearn.graphlearn import GraphLearnSampler
from graphlearn.localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar
import logging
logger = logging.getLogger(__name__)
import networkx as nx
from graphlearn.utils import draw
import eden.util.display as edraw
import eden
import traceback


'''
1. tell the sampler to use new GraphManager
'''
class UberSampler(GraphLearnSampler):
    def get_graphmanager(self):
        return lambda x,y: UberGraphManager(x,y,[2])


class UberGraphManager(graphtools.Graphmanager):
    '''
     since i should not repeat myself, i will just use as much as possible
     from the Graphmanager implementation.
    '''

    def core_substitution(self, orig_cip_graph, new_cip_graph):
        graph=graphtools.core_substitution( self._base_graph, orig_cip_graph ,new_cip_graph )
        return UberGraphManager( graph, self.vectorizer)
    # ok

    #def mark_median(self, inp='importance', out='is_good', estimator=None):
    # fine

    #def clean(self):
    # fine

    #def out(self):
    # fine

    #def postprocess(self,postprocessor):
    # fine, we just dont do postproc :)

    def base_graph(self):
        return self._base_graph


    def graph(self):
        g= nx.disjoint_union(self._base_graph, self._abstract_graph)
        for n,d in g:
            if 'contracted' in d:
                for e in d['contracted']:
                    g.add_edge( n, e, nesting=True)
        return g


    def __init__(self,graph,vectorizer, some_thickness_list):
        self.some_thickness_list=some_thickness_list
        self._base_graph=graph
        if len(graph) > 0:
            self._base_graph=vectorizer._edge_to_vertex_transform(self._base_graph)
        self.vectorizer=vectorizer
        self._abstract_graph= make_abstract(self._base_graph,self.vectorizer)

    def extract_core_and_interface(self, root,thickness, **args):
        return extract_cips(root,self, base_thickness_list= thickness,**args)


    def all_cips(self,**args):

        graph=self._base_graph
        cips = []
        for root_node in graph.nodes_iter():
            if 'edge' in graph.node[root_node]:
                continue
            cip_list = self.extract_core_and_interface(root_node,thickness=self.some_thickness_list,**args)
            if cip_list:
                cips.append(cip_list)
        return cips


    def random_cip(self,radius_list=None,thickness_list=None, **args):

        node = random.choice(self._base_graph.nodes())
        if 'edge' in self._base_graph.node[node]:
            node = random.choice(self._base_graph.neighbors(node))
            # random radius and thickness
        args['radius_list'] = [random.choice(radius_list)]
        args['thickness_list'] = [random.choice(thickness_list)]
        random_something= [random.choice(self.some_thickness_list)]
        return self.extract_core_and_interface(node,thickness=random_something, **args)








def check_and_draw(base_graph, abstr):
    '''
    :param base_graph: a base graph
    :param abstr: an abstract graph
    :return: check if EVERY node in base_graph is in any abstr.graph.node['contracted']
    '''
    nodeset = set([a for n, d in abstr.nodes(data=True) for a in d['contracted']])
    broken = []
    for n in base_graph.nodes():
        if n not in nodeset:
            broken.append(n)
            base_graph.node[n]['colo'] = .5
    if len(broken) > 0:
        print "FOUND SOMETHING BROKEN:"
        draw.set_ids(base_graph)
        base_graph.graph['info'] = 'failed to see these:%s' % str(broken)
        edraw.draw_graph(base_graph, vertex_label='id', vertex_color='colo', edge_label=None, size=20)
        for e, d in abstr.nodes(data=True):
            d['label'] = str(d.get('contracted', ''))
        edraw.draw_graph(abstr, vertex_label='label', vertex_color=None, edge_label=None, size=20)
        return False
    return True


def make_abstract(graph, vectorizer):
    '''
        graph should be the same expanded graph that we will feed to extract_cips later...
    '''

    if isinstance(graph, nx.DiGraph):
        graph = graph.to_undirected()

    graph2 = vectorizer._revert_edge_to_vertex_transform(graph)
    graph2 = arbitrary_graph_abstraction_function(graph2)
    graph2 = vectorizer._edge_to_vertex_transform(graph2)

    # find out to which abstract node the edges belong
    # finding out where the edge-nodes belong, because the contractor cant possibly do this
    getabstr = {contra: node for node, d in graph2.nodes(data=True) for contra in d.get('contracted', [])}

    for n, d in graph.nodes(data=True):
        if 'edge' in d:
            # if we have found an edge node...
            # lets see whos left and right of it:
            n1, n2 = graph.neighbors(n)
            # case1: ok those belong to the same gang so we most likely also belong there.
            if getabstr[n1] == getabstr[n2]:
                graph2.node[getabstr[n1]]['contracted'].add(n)

            # case2: neighbors belong to different gangs...
            else:
                blub = set(graph2.neighbors(getabstr[n1])) & set(graph2.neighbors(getabstr[n2]))
                for blob in blub:
                    if 'contracted' in graph2.node[blob]:
                        graph2.node[blob]['contracted'].add(n)
                    else:
                        graph2.node[blob]['contracted'] = set([n])
    return graph2

def arbitrary_graph_abstraction_function(graph):
    '''
    # the function needs to set a 'contracted' attribute to each node with a set of vertices that
    # are contracted.
    :param graph: any graph   .. what kind? expanded? which flags musst be set?
    :return: an abstract graph with node annotations that refer to the node ids it is contracting
    '''
    # annotate in node attribute 'type' the incident edges' labels
    labeled_graph = vertex_attributes.incident_edge_label(
        [graph], level=2, output_attribute='type', separator='.').next()
    # do contraction
    contracted_graph = contraction(
        [labeled_graph], contraction_attribute='type', modifiers=[], nesting=False).next()
    return contracted_graph





def extract_cips(node,
                 graphmanager,
                 base_thickness_list=None,
                 hash_bitmask=None,
                 mod_dict={},
                 **argz):
    '''
    :param node: node in the abstract graph
    :param abstract_graph:  the abstract graph expanded
    :param base_graph:  the underlying real graph
    :param abstract_radius: radius in abstract graph
    :param abstract_thickness: thickness in abstr
    :param base_thickness:  thickness for the base graph
    :return:  a  list of cips
    '''
    # if not filter(abstract_graph, node):
    #    return []
    # print 'ok1'

    abstract_graph=graphmanager.abstract_graph()
    base_graph=graphmanager.base_graph()
    vectorizer=graphmanager.vectorizer

    #abstract_thickness_list=thickness_list
    #abstract_radius_list=radius_list

    if 'hlabel' not in abstract_graph.node[abstract_graph.nodes()[0]]:
        vectorizer._label_preprocessing(abstract_graph)
    if 'hlabel' not in base_graph.node[base_graph.nodes()[0]]:
        vectorizer._label_preprocessing(base_graph)

    # on the abstract graph we use the normal extract cip stuff:
    abstract_cips = graphtools.extract_core_and_interface(node,
                                                          abstract_graph,
                                                          #radius_list=abstract_radius_list,
                                                          #thickness_list=abstract_thickness_list,
                                                          vectorizer=vectorizer,
                                                          hash_bitmask=hash_bitmask,
                                                          **argz)

    cips = []

    for acip in abstract_cips:

            # now we need to calculate the real cips:
            # the trick is to also use the normal extractor, but in order to do that we need
            # to collapse the 'core'

            # MERGE THE CORE OF THE ABSTRACT GRAPH IN THE BASE GRAPH
        mergeids = [base_graph_id for radius in range(
            acip.radius + 1) for abstract_node_id in acip.distance_dict.get(radius)
            for base_graph_id in abstract_graph.node[abstract_node_id]['contracted']]
        base_copy = base_graph.copy()

        # remove duplicates:
        mergeids = list(set(mergeids))

        for node in mergeids[1:]:
            graphtools.merge(base_copy, mergeids[0], node)

        # do cip extraction and calculate the real core hash
        # draw.graphlearn_draw(base_copy,size=20)

        # draw.draw_center(base_copy,mergeids[0],5,size=20)
        # print base_thickness_list,hash_bitmask

        base_level_cips = graphtools.extract_core_and_interface(mergeids[0],
                                                                base_copy,
                                                                radius_list=[0],
                                                                thickness_list=base_thickness_list,
                                                                vectorizer=vectorizer,
                                                                hash_bitmask=hash_bitmask,
                                                                **argz)

        core_hash = graphtools.graph_hash(base_graph.subgraph(mergeids), hash_bitmask=hash_bitmask)

        # print base_level_cips
        acip.core_nodes_count= len(mergeids)

        # now we have a bunch of base_level_cips and need to attach info from the abstract cip.
        for base_cip in base_level_cips:

            # we cheated a little with the core, so we need to undo our cheating
            whatever = base_cip.graph.copy()
            base_cip.graph = base_graph.subgraph(base_cip.graph.nodes() + mergeids).copy()

            for n in mergeids:
                base_cip.graph.node[n]['core'] = True

            for n, d in base_cip.graph.nodes(data=True):
                if 'core' not in d:
                    d['interface'] = True

                    d['distance_dependent_label'] = whatever.node[n]['distance_dependent_label']

            base_cip.core_hash = core_hash

            # merging cip info with the abstract graph
            base_cip.interface_hash = eden.fast_hash_4(base_cip.interface_hash,
                                                       acip.interface_hash,
                                                       get_mods(mod_dict, mergeids), 0,
                                                       hash_bitmask)

            base_cip.core_nodes_count = acip.core_nodes_count
            base_cip.radius = acip.radius
            base_cip.abstract_thickness = acip.thickness

            # i want to see what they look like :)
            base_cip.abstract_view = acip.graph
            cips.append(base_cip)
    return cips


'''
a mod_dict is a modification dictionary.
use get_mod_dict to make a dict of nodenumber:associated_hash
if the nodenumber is in the core, the hash gets added to the interfacehash.

'''
def get_mods(mod_dict, nodes):
    su = 0
    for n in nodes:
        if n in mod_dict:
            su += mod_dict[n]
    return su
# here we create the mod dict once we have a graph..

def get_mod_dict(graph):
    return {}
