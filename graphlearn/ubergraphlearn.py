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
from eden.util.display import draw_graph
'''
first we build the new sampler that is able to handle abstract graphs...
'''

class UberSampler(GraphLearnSampler):

    def __init__(self,real_thickness_list=[1,2,3],grammar=None,**kwargs):
        '''
            graphlernsampler with its extensions..

            for now this:
                is a real_thickness_list
                and we make sure that the grammar can handle our new corez :)
        '''
        if grammar:
            assert isinstance(grammar,UberGrammar)
        self.real_thickness_list=[2*e for e in real_thickness_list]
        super(UberSampler, self).__init__(grammar,**kwargs)


    def fit_grammar(self, graphs, core_interface_pair_remove_threshold=2, interface_remove_threshold=2, n_jobs=-1):
        if not self.local_substitutable_graph_grammar:
            self.local_substitutable_graph_grammar = UberGrammar(
                self.radius_list,
                self.thickness_list,
                complexity=self.complexity,
                core_interface_pair_remove_threshold=core_interface_pair_remove_threshold,
                interface_remove_threshold=interface_remove_threshold,
                nbit=self.nbit,
                node_entity_check=self.node_entity_check,
                real_thickness_list=self.real_thickness_list)
        self.local_substitutable_graph_grammar.fit(graphs, n_jobs)


    def  _original_cip_extraction(self,graph):
        '''
        selects the next candidate.
        '''
        abstr= make_abstract(graph)
        node = random.choice(abstr.nodes())
        if 'edge' in abstr.node[node]:
            node = random.choice(abstr.neighbors(node))
            # random radius and thickness
        radius = random.choice(self.radius_list)
        thickness = random.choice(self.thickness_list)
        real_thickness = random.choice(self.real_thickness_list)

        return extract_cips(node,abstr, graph, [radius], [thickness],[real_thickness], vectorizer=self.vectorizer,
                                             hash_bitmask=self.hash_bitmask, filter=self.node_entity_check)


'''
 here we build the new grammar things.. we basically just alter how graphs are read.
'''

class UberGrammar(LocalSubstitutableGraphGrammar):

    def argbuilder(self, graphs, batch_size=10):
        args = [self.radius_list, self.thickness_list, self.vectorizer, self.hash_bitmask, self.node_entity_check, self.real_thickness_list]
        function = extract_cores_and_interfaces
        for batch in grouper(graphs, batch_size):
            yield dill.dumps((function, args, batch))

    def __init__(self,real_thickness_list=None,**kwargs):
        self.real_thickness_list=real_thickness_list
        super(UberSampler, self).__init__(**kwargs)


def extract_cores_and_interfaces(parameters):
    # happens if batcher fills things up with null
    if parameters[0] is None:
        return None
    try:
        # unpack arguments, expand the graph
        graph, radius_list, thickness_list, vectorizer, hash_bitmask, node_entity_check , real_thickness_list = parameters
        graph = vectorizer._edge_to_vertex_transform(graph)
        cips = []
        abstr= make_abstract(graph,vectorizer)

        for node in abstr.nodes_iter():
            if 'edge' in graph.node[node]:
                continue
            core_interface_list = extract_cips(
                node,
                abstr ,
                graph,
                radius_list,
                thickness_list,
                real_thickness_list,
                vectorizer=vectorizer,
                hash_bitmask=hash_bitmask,
                filter=node_entity_check)
            if core_interface_list:
                cips.append(core_interface_list)
        return cips
    except:
        logger.info( "extract_cores_and_interfaces_died" )
        logger.info( parameters )


'''
the things down here replace functions in the graphtools.
'''


def get_abstraction(graph):
    '''
    :param graph: any graph   .. what kind? expanded? which flags musst be set?
    :return: an abstract graph with node annotations that refer to the node ids it is contracting
    '''
    #annotate in node attribute 'type' the incident edges' labels

    graph = vertex_attributes.incident_edge_label(
        [graph], level = 2, output_attribute = 'type', separator = '.').next()
    graph = contraction(
        [graph], contraction_attribute = 'type', modifiers = [], nesting = False).next()
    return graph


def make_abstract(graph,vectorizer):
    '''
        graph should be the same expanded graph that we will feed to extract_cips later...
    '''
    graph2 = vectorizer._revert_edge_to_vertex_transform (graph)
    graph2 = get_abstraction(graph2)
    graph2.graph.pop('expanded') # EDEN WORKAROUND !!!!!!!!!
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



    '''  i let this here.. in case you ever need to display what this function does..
    draw.set_ids(graph)
    draw_graph(graph, vertex_label='id',vertex_color='colo', edge_label=None,size=20)

    for e,d in graph2.nodes(data=True):
        d['label']=str(d['contracted'])
    draw_graph(graph2, vertex_label='label',vertex_color=None, edge_label=None,size=20)
    '''
    return graph2


def extract_cips(node,
    abstract_graph, real_graph ,abstract_radius_list,abstract_thickness_list, real_thickness_list,vectorizer,**argz):
    '''
    :param node: node in the abstract graph
    :param abstract_graph:  the abstract graph expanded
    :param real_graph:  the underlying real graph
    :param abstract_radius: radius in abstract graph
    :param abstract_thickness: thickness in abstr
    :param real_thickness:  thickness in real graph
    :return:  a  list of cips
    '''
    #if not filter(abstract_graph, node):
    #    return []
    if 'hlabel' not in abstract_graph.node[0]:
        vectorizer._label_preprocessing(abstract_graph)
    if 'hlabel' not in real_graph.node[0]:
        vectorizer._label_preprocessing(real_graph)
    # argz shoud be this stuff:
    #vectorizer=None, filter=lambda x, y: True, hash_bitmask
    abstract_cips=graphtools.extract_core_and_interface(node,
        abstract_graph, radius_list=abstract_radius_list, thickness_list=abstract_thickness_list,vectorizer=vectorizer,**argz)


    draw.display(abstract_cips[0].graph, vertex_label='id',size=10)



    cips=[]
    for acip in abstract_cips:

            # MERGE THE CORE TO A SINGLE NODE::
            mergeids = [   abstract_graph.node[n]['contracted']  for z in range(acip.radius+1)  for n in acip.distance_dict.get(z)  ]
            mergeids = [id for sublist in mergeids for id in sublist ]
            real_copy= nx.Graph(real_graph)


            print mergeids
            for m in mergeids:
                real_copy.node[m]['colo']=0.5
            draw.set_ids(real_copy)
            draw_graph(real_copy, vertex_label='id',vertex_color='colo', edge_label=None,size=20)


            for e,d in acip.graph.nodes(data=True):
                d['label']=str(d['contracted'])
            draw_graph(acip.graph,vertex_label='label')


            for node in mergeids[1:]:
                graphtools.merge(real_copy,mergeids[0],node)



            # BECAUSE WE COLLAPSED THE CORE WE CAN USE THE NORMAL EXTRACTOR AGAIM
            lowlevelcips = graphtools.extract_core_and_interface(mergeids[0],
                real_copy,radius_list=[0],thickness_list=real_thickness_list,vectorizer=vectorizer,**argz)


            for lowcip in lowlevelcips:

                #build real graph,  reverting the merge
                lowcip.graph=real_graph.subgraph(lowcip.graph.nodes()+mergeids)

                # of course we also need to mark the core nodes...
                for n in mergeids:
                    lowcip.graph.node[n]['core']=True

                #the hash needs to me melted with the abstract one OoeOO
                lowcip.interface_hash+=acip.interface_hash

                # core hash needs to be 'correct'
                # not sure  which one i should use..
                lowcip.core_hash= acip.core_hash

                #corecount
                lowcip.core_nodes_count = acip.core_nodes_count
                cips.append(lowcip)
    return cips