import graphlearn.abstract_graphs.rnaabstract
from ubergraphlearn import UberSampler,UberGrammar
import ubergraphlearn
import networkx as nx
import graphlearn.utils.draw as draw

import directedgraphtools as dgtools

class RNASampler(UberSampler):

    def _propose(self, graph):
        '''
         we wrap the propose single cip, so it may be overwritten some day

         why am i doing this? need to see if deleteable
        '''
        graph2 = None
        while graph2 == None:
            graph2 = self._propose_graph(graph)
        return graph2

    '''
    def _stop_condition(self, graph):
        self.last_graph=graph.copy()
        if not is_rna(graph):
            self._sample_path=[self.last_graph]
            raise Exception('WE CREATED THE ANTI RNA')
    '''

    '''
        turning sample starter graph to digraph
    '''
    def _sample_init(self, graph):
        #graph = self.vectorizer._edge_to_vertex_transform(graph)
        #graph = dgtools.expanded_rna_graph_to_digraph(graph)
        self.postprocessor.fit(self)

        graph=self.postprocessor.postprocess(graph)
        if graph is None:
            raise Exception ('_sample_init failed, cant fold to trna')
        self._score(graph)
        self._sample_notes = ''
        self._sample_path_score_set = set()
        return graph

    def _score(self,graph):

        estimateable=graph.graphmanager.get_estimateable()
        super(RNASampler,self)._score(estimateable)
        graph._score=estimateable._score
        return graph._score


    '''
        this is also used sometimes so we make better sure it doesnt fail
    '''
    def _revert_edge_to_vertex_transform(self,graph):
        # making it to a normal graph before we revert
        graph=nx.Graph(graph)
        try:
            graph=self.vectorizer._revert_edge_to_vertex_transform(graph)
            return graph
        except:
            print 'rnasampler: revert edge to vertex transform failed'
            draw.graphlearn_draw(graph,contract=False, size=20)



    def __init__(self,**kwargs):
        super(RNASampler, self).__init__(**kwargs)
        self.feasibility_checker.checklist.append(is_rna)


'''
    rna checker
'''
def is_rna (graph):
    graph=graph.copy()
    # remove structure
    bonds= [ n for n,d in graph.nodes(data=True) if d['label']=='=' ]
    graph.remove_nodes_from(bonds)
    # see if we are cyclic
    for node,degree in graph.in_degree_iter( graph.nodes() ):
        if degree == 0:
            break
    else:
        return False
    # check if we are connected.
    graph=nx.Graph(graph)
    return nx.is_connected(graph)





# modifying  ubergraphlearn further..
def get_mod_dict(graph):
    s,e= graphlearn.abstract_graphs.rnaabstract.get_start_and_end_node(graph)
    return {s:696969 , e:123123123}
#ubergraphlearn.get_mod_dict=get_mod_dict
import rnaabstract


#ubergraphlearn.make_abstract = rnaabstract.direct_abstractor
#ubergraphlearn.make_abstract = rnaabstract.direct_abstraction_wrapper
