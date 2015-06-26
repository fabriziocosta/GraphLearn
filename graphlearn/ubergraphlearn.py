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
        super(UberSampler, self).__init__(grammar=grammar,**kwargs)


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


        #edraw.draw_graph(g[0].graph,edge_label=None,size=20)
        return g


'''
 here we build the new grammar things.. we basically just alter how graphs are read.
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


    graph = vertex_attributes.incident_edge_label(
        [graph], level = 2, output_attribute = 'type', separator = '.').next()

    '''
    print "DEBUGOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR mazu getabstr"
    graph2= graph.copy()
    graph2 = contraction(
        [graph2], contraction_attribute = 'type', modifiers = [], nesting = True).next()
    edraw.draw_graph(graph2, vertex_label='label',vertex_color=None, edge_label=None,size=30)
    '''

    graph = contraction(
        [graph], contraction_attribute = 'type', modifiers = [], nesting = False).next()

    return graph


def make_abstract(graph,vectorizer):
    '''
        graph should be the same expanded graph that we will feed to extract_cips later...
    '''
    graph2 = vectorizer._revert_edge_to_vertex_transform (graph)
    #g3=graph2
    graph2 = arbitrary_graph_abstraction_function(graph2)
    #g4=graph2
    #graph2.graph.pop('expanded') # EDEN WORKAROUND !!!!!!!!!
    graph2 = vectorizer._edge_to_vertex_transform (graph2)

    '''
    draw.set_ids(g3)
    edraw.draw_graph(g3, vertex_label='id',vertex_color='colo', edge_label=None,size=20)

    for e,d in g4.nodes(data=True):
        d['label']=str(d.get('contracted',''))
    edraw.draw_graph(g4, vertex_label='label',vertex_color=None, edge_label=None,size=20)
    '''


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



    '''#  i let this here.. in case you ever need to display what this function does..
    draw.set_ids(graph)
    edraw.draw_graph(graph, vertex_label='id',vertex_color='colo', edge_label=None,size=20)

    for e,d in graph2.nodes(data=True):
        d['label']=str(d['contracted'])
    edraw.draw_graph(graph2, vertex_label='label',vertex_color=None, edge_label=None,size=20)'''

    return graph2


def extract_cips(node,
    abstract_graph, base_graph ,abstract_radius_list=None,abstract_thickness_list=None, base_thickness_list=None,vectorizer=None,hash_bitmask=None,**argz):
    '''
    :param node: node in the abstract graph
    :param abstract_graph:  the abstract graph expanded
    :param base_graph:  the underlying real graph
    :param abstract_radius: radius in abstract graph
    :param abstract_thickness: thickness in abstr
    :param base_thickness:  thickness in real graph
    :return:  a  list of cips
    '''
    #if not filter(abstract_graph, node):
    #    return []
    if 'hlabel' not in abstract_graph.node[0]:
        vectorizer._label_preprocessing(abstract_graph)
    if 'hlabel' not in base_graph.node[0]:
        vectorizer._label_preprocessing(base_graph)
    # argz shoud be this stuff:
    #vectorizer=None, filter=lambda x, y: True, hash_bitmask
    abstract_cips=graphtools.extract_core_and_interface(node,
        abstract_graph, radius_list=abstract_radius_list, thickness_list=abstract_thickness_list,vectorizer=vectorizer,hash_bitmask=hash_bitmask,**argz)


    #draw.display(abstract_cips[0].graph, vertex_label='id',size=10)


    cips=[]
    for acip in abstract_cips:

            # MERGE THE CORE TO A SINGLE NODE:
            mergeids = [   abstract_graph.node[n]['contracted']  for z in range(acip.radius+1)  for n in acip.distance_dict.get(z)  ]
            mergeids = [id for sublist in mergeids for id in sublist ]



            base_copy= nx.Graph(base_graph)

            ''' you will see the abstract interface with its nesting nodes on the real graph and
                the real graph with the core marked
            print mergeids
            for m in mergeids:
                base_copy.node[m]['colo']=0.5
            draw.set_ids(base_copy)
            #draw_graph(base_copy, vertex_label='id',vertex_color='colo', edge_label=None,size=20)

            for e,d in acip.graph.nodes(data=True):
                d['id']=str(d['contracted'])
            #edraw.draw_graph(acip.graph,vertex_label='id')
            edraw.draw_graph_set([base_copy,acip.graph], vertex_label='id',vertex_color='colo', edge_label=None,size=20)
            '''
            for node in mergeids[1:]:
                graphtools.merge(base_copy,mergeids[0],node)

            #draw.draw_center(base_copy,mergeids[0],5)

            # BECAUSE WE COLLAPSED THE CORE WE CAN USE THE NORMAL EXTRACTOR AGAIM
            base_level_cips = graphtools.extract_core_and_interface(mergeids[0],
                base_copy,radius_list=[0],thickness_list=base_thickness_list,vectorizer=vectorizer,hash_bitmask=hash_bitmask,**argz)

            core_hash= graphtools.calc_core_hash(base_graph.subgraph(mergeids),hash_bitmask=hash_bitmask)

            for base_cip in base_level_cips:

                #build real graph,  reverting the merge
                base_cip.graph=nx.Graph(base_graph.subgraph(base_cip.graph.nodes()+mergeids))

                # of course we also need to mark the core nodes...
                for n in mergeids:
                    base_cip.graph.node[n]['core']=True

                #the hash needs to me melted with the abstract one OoeOO
                base_cip.interface_hash+=acip.interface_hash

                # core hash needs to be 'correct'
                # not sure  which one i should use..
                base_cip.core_hash= core_hash

                #corecount
                base_cip.core_nodes_count = acip.core_nodes_count

                base_cip.radius=acip.radius
                base_cip.abstract_thickness= acip.thickness

                cips.append(base_cip)


    return cips