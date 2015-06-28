from eden.modifier.graph import vertex_attributes
from eden.modifier.graph.structure import contraction
import graphtools
import random
from graphlearn import GraphLearnSampler
from localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar
import logging
logger = logging.getLogger(__name__)
import dill
from eden import grouper
import networkx as nx
from utils import draw
import eden.util.display as edraw
import eden
import traceback
'''
first we build the new sampler that is able to handle abstract graphs...
'''

class UberSampler(GraphLearnSampler):

    def __init__(self,base_thickness_list=[1,2,3],core_interface_pair_remove_threshold=1, interface_remove_threshold=2,grammar=None,**kwargs):
        '''
            graphlernsampler with its extensions..

            for now this:
                is a real_thickness_list
                and we make sure that the grammar can handle our new corez :)
        '''
        # if we get a grammar we make sure that it is a ubergrammar
        if grammar:
            assert isinstance(grammar,UberGrammar)

        self.base_thickness_list=[int(2*e) for e in base_thickness_list]
        super(UberSampler, self).__init__(grammar=grammar,core_interface_pair_remove_threshold=core_interface_pair_remove_threshold, interface_remove_threshold= interface_remove_threshold,**kwargs)


        # after the normal run, a grammar was created, but its a ordinary grammar .. so we build a new one
        if not isinstance(self.local_substitutable_graph_grammar,UberGrammar):
            self.local_substitutable_graph_grammar = UberGrammar(
                base_thickness_list=self.base_thickness_list,
                radius_list=self.radius_list,
                thickness_list=self.thickness_list,
                complexity=self.complexity,
                core_interface_pair_remove_threshold=core_interface_pair_remove_threshold,
                interface_remove_threshold=interface_remove_threshold,
                nbit=self.nbit,
                node_entity_check=self.node_entity_check)


    def  _original_cip_extraction(self,graph):
        '''
        selects the next candidate.
        '''
        graph=self.vectorizer._edge_to_vertex_transform(graph)
        abstr= make_abstract(graph,self.vectorizer)
        node = random.choice(abstr.nodes())
        if 'edge' in abstr.node[node]:
            node = random.choice(abstr.neighbors(node))
        # random radius and thickness
        radius = random.choice(self.radius_list)
        thickness = random.choice(self.thickness_list)
        base_thickness = random.choice(self.base_thickness_list)

        g= extract_cips(node,abstr, graph, [radius], [thickness],[base_thickness], vectorizer=self.vectorizer,
                                             hash_bitmask=self.hash_bitmask, filter=self.node_entity_check)


        return g


'''
 here we adjust the grammar.
'''

class UberGrammar(LocalSubstitutableGraphGrammar):

    def _multi_process_argbuilder(self, graphs, batch_size=10):
        args = [self.radius_list, self.thickness_list, self.vectorizer, self.hash_bitmask, self.node_entity_check, self.base_thickness_list]
        function = extract_cores_and_interfaces_mk2
        for batch in grouper(graphs, batch_size):
            yield dill.dumps((function, args, batch))

    def __init__(self,base_thickness_list=None,**kwargs):
        self.base_thickness_list=base_thickness_list
        super(UberGrammar, self).__init__(**kwargs)

    def _read_single(self, graphs):
        """
            for graph in graphs:
                get cips of graph
                    put cips into grammar
        """
        for gr in graphs:
            problem = (
                gr, self.radius_list, self.thickness_list, self.vectorizer, self.hash_bitmask, self.node_entity_check,self.base_thickness_list)
            for core_interface_data_list in extract_cores_and_interfaces_mk2(problem):
                for cid in core_interface_data_list:
                    self._add_core_interface_data(cid)


def extract_cores_and_interfaces_mk2(parameters):
    # happens if batcher fills things up with null
    if parameters[0] is None:
        return None
    try:
        # unpack arguments, expand the graph
        graph, radius_list, thickness_list, vectorizer, hash_bitmask, node_entity_check , base_thickness_list = parameters
        graph = vectorizer._edge_to_vertex_transform(graph)
        cips = []
        abstr= make_abstract(graph,vectorizer)

        for node in abstr.nodes_iter():
            if 'edge' in abstr.node[node]:
                continue
            core_interface_list = extract_cips(
                node,
                abstr ,
                graph,
                radius_list,
                thickness_list,
                base_thickness_list,
                vectorizer=vectorizer,
                hash_bitmask=hash_bitmask,
                filter=node_entity_check)
            if core_interface_list:
                cips.append(core_interface_list)
        return cips

    except Exception as exc:
            logger.info(exc)
            logger.info(traceback.format_exc(10))
            logger.info( "extract_cores_and_interfaces_died" )
            logger.info( parameters )



'''
the things down here replace functions in the graphtools.
'''


def is_rna (graph):
    endcount=0
    for n,d in graph.nodes(data=True):
        if d['node']==True:
            neighbors=graph.neighbors(n)
            backbonecount= len( [ 1 for n in neighbors if graph.node[n]['label']=='-' ] )
            if backbonecount == 2:
                continue
            if backbonecount == 1:
                endcount+=1
            if backbonecount > 2:
                raise Exception ('backbone broken')
    return endcount == 2

def arbitrary_graph_abstraction_function(graph):
    '''
    # the function needs to set a 'contracted' attribute to each node with a set of vertices that are contractet.
    :param graph: any graph   .. what kind? expanded? which flags musst be set?
    :return: an abstract graph with node annotations that refer to the node ids it is contracting
    '''

    #annotate in node attribute 'type' the incident edges' labels
    labeled_graph = vertex_attributes.incident_edge_label(
        [graph], level = 2, output_attribute = 'type', separator = '.').next()
    # do contraction
    contracted_graph = contraction(
        [labeled_graph], contraction_attribute = 'type', modifiers = [], nesting = False).next()
    return contracted_graph


def check_and_draw(base_graph,abstr):
    '''
    :param base_graph: a base graph
    :param abstr: an abstract graph
    :return: check if EVERY node in base_graph is in any abstr.graph.node['contracted']
    '''
    nodeset= set(   [ a for n,d in abstr.nodes(data=True) for a in d['contracted'] ]  )
    broken=[]
    for n in base_graph.nodes():
        if n not in nodeset:
            broken.append(n)
            base_graph.node[n]['colo']=.5
    if len(broken)>0:
        print "FOUND SOMETHING BROKEN:"
        draw.set_ids(base_graph)
        base_graph.graph['info']='failed to see these:%s' % str(broken)
        edraw.draw_graph(base_graph, vertex_label='id',vertex_color='colo', edge_label=None,size=20)
        for e,d in abstr.nodes(data=True):
            d['label']=str(d.get('contracted',''))
        edraw.draw_graph(abstr, vertex_label='label',vertex_color=None, edge_label=None,size=20)
        return False
    return True


def make_abstract(graph,vectorizer):
    '''
        graph should be the same expanded graph that we will feed to extract_cips later...
    '''
    graph2 = vectorizer._revert_edge_to_vertex_transform (graph)
    graph2 = arbitrary_graph_abstraction_function(graph2)
    graph2 = vectorizer._edge_to_vertex_transform (graph2)

    # find out to which abstract node the edges belong
    # finding out where the edge-nodes belong, because the contractor cant possibly do this
    getabstr={ contra:node for node,d in graph2.nodes(data=True) for contra in d.get('contracted',[])  }

    for n,d in graph.nodes(data=True):
        if 'edge' in d:
            # if we have found an edge node...
            # lets see whos left and right of it:
            n1,n2=graph.neighbors(n)
            # case1: ok those belong to the same gang so we most likely also belong there.
            if getabstr[n1]==getabstr[n2]:
                graph2.node[getabstr[n1]]['contracted'].add(n)

            # case2: neighbors belong to different gangs...
            else:
                 blub = set( graph2.neighbors( getabstr[n1])) & set( graph2.neighbors(getabstr[n2]))
                 for blob in blub:
                    if 'contracted' in graph2.node[blob]:
                        graph2.node[blob]['contracted'].add(n)
                    else:
                        graph2.node[blob]['contracted']=set([n])

    return graph2


def extract_cips(node,
    abstract_graph, base_graph ,abstract_radius_list=None,abstract_thickness_list=None, base_thickness_list=None,vectorizer=None,hash_bitmask=None,**argz):
    '''
    :param node: node in the abstract graph
    :param abstract_graph:  the abstract graph expanded
    :param base_graph:  the underlying real graph
    :param abstract_radius: radius in abstract graph
    :param abstract_thickness: thickness in abstr
    :param base_thickness:  thickness for the base graph
    :return:  a  list of cips
    '''
    #if not filter(abstract_graph, node):
    #    return []
    if 'hlabel' not in abstract_graph.node[ abstract_graph.nodes()[0] ]:
        vectorizer._label_preprocessing(abstract_graph)
    if 'hlabel' not in base_graph.node[ base_graph.nodes()[0] ]:
        vectorizer._label_preprocessing(base_graph)


    # on the abstract graph we use the normal extract cip stuff:
    abstract_cips=graphtools.extract_core_and_interface(node,
        abstract_graph, radius_list=abstract_radius_list, thickness_list=abstract_thickness_list,vectorizer=vectorizer,hash_bitmask=hash_bitmask,**argz)


    cips=[]
    for acip in abstract_cips:

            # now we need to calculate the real cips:
            # the trick is to also use the normal extractor, but in order to do that we need to collapse the 'core'

            # MERGE THE CORE OF THE ABSTRACT GRAPH IN THE BASE GRAPH
            mergeids = [ base_graph_id for radius in range(acip.radius+1) for abstract_node_id in acip.distance_dict.get(radius) for base_graph_id in abstract_graph.node[abstract_node_id]['contracted']  ]
            base_copy= nx.Graph(base_graph)
            for node in mergeids[1:]:
                graphtools.merge(base_copy,mergeids[0],node)

            # do cip extraction and calculate the real core hash
            base_level_cips = graphtools.extract_core_and_interface(mergeids[0],
                base_copy,radius_list=[0],thickness_list=base_thickness_list,vectorizer=vectorizer,hash_bitmask=hash_bitmask,**argz)
            core_hash= graphtools.calc_core_hash(base_graph.subgraph(mergeids),hash_bitmask=hash_bitmask)

            # now we have a bunch of base_level_cips and need to attach info from the abstract cip.
            for base_cip in base_level_cips:

                # we cheated a little with the core, so we need to undo our cheating
                base_cip.graph=nx.Graph(base_graph.subgraph(base_cip.graph.nodes()+mergeids))
                for n in mergeids:
                    base_cip.graph.node[n]['core']=True
                base_cip.core_hash= core_hash

                # merging cip info with the abstract graph
                base_cip.interface_hash=eden.fast_hash_2(base_cip.interface_hash,acip.interface_hash,hash_bitmask)
                base_cip.core_nodes_count = acip.core_nodes_count
                base_cip.radius=acip.radius
                base_cip.abstract_thickness= acip.thickness
                cips.append(base_cip)
    return cips