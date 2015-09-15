from eden.modifier.graph import vertex_attributes
from eden.modifier.graph.structure import contraction
import graphlearn.graphtools as graphtools
from graphlearn.graphtools import GraphWrapper
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




class UberGraphWrapper(GraphWrapper):
    '''
     since i should not repeat myself, i will just use as much as possible
     from the Graphmanager implementation.
    '''
    def __str__(self):
        #return 'Ubermanager: base_nodes: %d abstract_nodes: %d' % (len(self._base_graph),len(self._abstract_graph))
        return '\n'.join (['%s:%s'% (str(k),v) for k,v in self.__dict__.items()]+[str(len(self._base_graph)),str(len(self._abstract_graph))])



    def core_substitution(self, orig_cip_graph, new_cip_graph):
        graph=graphtools.core_substitution( self._base_graph, orig_cip_graph ,new_cip_graph )
        return self.__class__( graph, self.vectorizer , self.some_thickness_list)
    # ok

    #def mark_median(self, inp='importance', out='is_good', estimator=None):
    # fine

    #def clean(self):
    #    graphtools.graph_clean(self._base_graph)
    #    graphtools.graph_clean(self._abstract_graph)

    #def out(self):
    # fine

    #def postprocess(self,postprocessor):
    # fine, we just dont do postproc :)

    #def base_graph(self):
    # fine


    def graph(self, nested=False):
        #draw.graphlearn_draw([self._base_graph, self._abstract_graph],size=20)
        g= nx.disjoint_union(self._base_graph, self.abstract_graph())

        if nested:
            for n,d in g.nodes(data=True):
                if 'contracted' in d:
                    for e in d['contracted']:
                        g.add_edge( n, e, nesting=True)
        return g



    def abstract_graph(self):
        if self._abstract_graph== None:
            self._abstract_graph = make_abstract(self._base_graph,self.vectorizer)
        return self._abstract_graph

    def __init__(self,graph,vectorizer=eden.graph.Vectorizer(), base_thickness_list=None):
        self.some_thickness_list=base_thickness_list
        self._base_graph=graph
        if len(graph) > 0:
            self._base_graph=vectorizer._edge_to_vertex_transform(self._base_graph)
        self.vectorizer=vectorizer
        self._abstract_graph= None

    def rooted_core_interface_pairs(self, root,thickness = None , **args):
        if thickness==None:
            thickness=self.some_thickness_list
        return extract_cips(root,self, base_thickness_list= thickness,**args)


    def all_core_interface_pairs(self,**args):
        graph=self.abstract_graph()
        cips = []
        for root_node in graph.nodes_iter():
            if 'edge' in graph.node[root_node]:
                continue
            cip_list = self.rooted_core_interface_pairs(root_node,**args)
            if cip_list:
                cips.append(cip_list)
        return cips


    def random_core_interface_pair(self,radius_list=None,thickness_list=None, **args):
        node = random.choice(self.abstract_graph().nodes())
        if 'edge' in self._abstract_graph.node[node]:
            node = random.choice(self._abstract_graph.neighbors(node))
            # random radius and thickness
        args['radius_list'] = [random.choice(radius_list)]
        args['thickness_list'] = [random.choice(thickness_list)]
        random_something= [random.choice(self.some_thickness_list)]
        return self.rooted_core_interface_pairs(node,thickness=random_something, **args)





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
    graph2 = edge_type_in_radius_abstraction(graph2)
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

def edge_type_in_radius_abstraction(graph):
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



def score_kmeans_abstraction(graph,score_attribute='importance',group='class'):
    '''
    :graph: nonexpanded graphs
    :score_attribute: see below, importance is default output of eden
    :group: take something that i can overwrite :)
    :returns: contracted graph

    this is not used currently, but i thought the idea was good...

    i need an annotated graph.. this may look like this:
    graph2 = self._base_graph.copy()  # annotate kills the graph i assume
    graph2 = self.vectorizer.annotate([graph2], estimator=estimator).next()
    '''

    # get values
    # ill need a list of lists for kmeans input
    values = []
    for n, d in graph.nodes(data=True):
        if 'edge' not in d:
            values.append( [d[score_attribute]])

    #3 means oO
    from sklearn.cluster import KMeans
    est=KMeans(n_clusters=3)
    est.fit(values)

    # mark class
    # again we need to make the attribute to a list for kmeans
    # result of predict is some numpy stuff, we want python ints
    for n, d in graph.nodes(data=True):
        if 'edge' not in d:
            d[group]= int(est.predict( [d[score_attribute]]))

    # contract and return
    return contraction(
        [graph], contraction_attribute=group, modifiers=[], nesting=False).next()


def extract_cips(node,
                 graphmanager,
                 base_thickness_list=None,
                 hash_bitmask=None,
                 mod_dict={},
                 **argz):
    '''
    :param node: node in the abstract graph
    ::
    :return:  a  list of cips
    '''
    # if not filter(abstract_graph, node):
    #    return []

    #PREPARE
    abstract_graph=graphmanager.abstract_graph()
    base_graph=graphmanager.base_graph()
    vectorizer=graphmanager.vectorizer
    if 'hlabel' not in abstract_graph.node[abstract_graph.nodes()[0]]:
        vectorizer._label_preprocessing(abstract_graph)
    if 'hlabel' not in base_graph.node[base_graph.nodes()[0]]:
        vectorizer._label_preprocessing(base_graph)

    # EXTRACT CIPS NORMALY ON ABSTRACT GRAPH
    abstract_cips = graphtools.extract_core_and_interface(node,
                                                          abstract_graph,
                                                          vectorizer=vectorizer,
                                                          hash_bitmask=hash_bitmask,
                                                          **argz)


    # VOR EVERY ABSTRACT CIP: MERGE CORE IN BASE GRAPH AND APPLY CIP EXTRACTON
    cips = []
    for abstract_cip in abstract_cips:
        base_copy, mergeids = merge_core(base_graph.copy(),abstract_graph,abstract_cip)
        argz['thickness_list'] = base_thickness_list
        argz['radius_list'] = [0]
        base_level_cips = graphtools.extract_core_and_interface(mergeids[0],
                                                                base_copy,
                                                                vectorizer=vectorizer,
                                                                hash_bitmask=hash_bitmask,
                                                                **argz)


        # VOR EVERY BASE CIP: RESTORE CORE  AND  MERGE INFORMATION WITH ABSTRACT CIP
        core_hash = graphtools.graph_hash(base_graph.subgraph(mergeids), hash_bitmask=hash_bitmask)
        abstract_cip.core_nodes_count= len(mergeids)
        for base_cip in base_level_cips:
            cips.append(enhance_base_cip(base_cip, abstract_cip,mergeids,base_graph,hash_bitmask,mod_dict,core_hash))


    return cips


def enhance_base_cip(base_cip, abstract_cip,mergeids,base_graph,hash_bitmask,mod_dict,core_hash):
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
                                                   abstract_cip.interface_hash,
                                                   get_mods(mod_dict, mergeids), 0,
                                                   hash_bitmask)

        base_cip.core_nodes_count = abstract_cip.core_nodes_count
        base_cip.radius = abstract_cip.radius
        base_cip.abstract_thickness = abstract_cip.thickness

        # i want to see what they look like :)
        base_cip.abstract_view = abstract_cip.graph

        return base_cip






def merge_core(base_graph,abstract_graph,abstract_cip):
    """

    :param base_graph: base graph. will be consumed
    :param abstract_graph:  we want the contracted info.. maybe we also find this in the cip.. not sure
    :param abstract_cip: the abstract cip
    :return: we merge all the nodes in the base_graph, that belong to the core of the abstract_cip
    """

    mergeids = [base_graph_id for radius in range(
        abstract_cip.radius + 1) for abstract_node_id in abstract_cip.distance_dict.get(radius)
        for base_graph_id in abstract_graph.node[abstract_node_id]['contracted']]
    base_copy = base_graph.copy()

    # remove duplicates:
    mergeids = list(set(mergeids))

    for node_id in mergeids[1:]:
        graphtools.merge(base_copy, mergeids[0], node_id)

    return base_graph,mergeids









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
