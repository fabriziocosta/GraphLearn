
from ubergraphlearn import UberSampler,UberGrammar
import ubergraphlearn
import networkx as nx
import graphlearn.utils.draw as draw
import random
from eden.converter.rna.rnafold import rnafold_to_eden
import directedgraphtools

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
        graph = self.vectorizer._edge_to_vertex_transform(graph)
        graph = expanded_rna_graph_to_digraph(graph)
        self._score(graph)
        self._sample_notes = ''
        self._sample_path_score_set = set()
        self.postprocessor.fit(self)
        return graph


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



class PostProcessor:


    def __init__(self):
        pass

    def fit(self, other):
        print 'OMG i got a vectorizer kthx'
        self.vectorizer=other.vectorizer

    def postprocess(self, graph):
        return self.rna_refold( graph )

    def rna_refold(self, digraph=None, seq=None,vectorizer=None):
        """
        :param digraph:
        :param seq:

        :return: will extract a sequence, RNAfold it and create a abstract graph
        """
        # get a sequence no matter what :)
        if not seq:
            seq= rnaabstract.get_sequence(digraph)

        #print 'seq:',seq

        graph = rnafold_to_eden([('emptyheader',seq)], shape_type=5, energy_range=30, max_num=3).next()

        expanded_graph = self.vectorizer._edge_to_vertex_transform(graph)
        ex_di_graph = expanded_rna_graph_to_digraph(expanded_graph)
        #abstract_graph = directedgraphtools.direct_abstraction_wrapper(graph,0)
        return ex_di_graph



def expanded_rna_graph_to_digraph(graph):
    '''
    :param graph:  an expanded rna representing graph as produced by eden.
                   properties: backbone edges are replaced by a node labeled '-'.
                   rna reading direction is reflected by ascending node ids in the graph.
    :return: a graph, directed edges along the backbone
    '''
    digraph=nx.DiGraph(graph)
    for n,d in digraph.nodes(data=True):
        if 'edge' in d:
            if d['label']=='-':
                ns=digraph.neighbors(n)
                ns.sort()
                digraph.remove_edge(ns[1],n)
                digraph.remove_edge(n,ns[0])
    return digraph



# modifying  ubergraphlearn further..
def get_mod_dict(graph):
    s,e=directedgraphtools.get_start_and_end_node(graph)
    return {s:696969 , e:123123123}
ubergraphlearn.get_mod_dict=get_mod_dict
import rnaabstract


#ubergraphlearn.make_abstract = rnaabstract.direct_abstractor
ubergraphlearn.make_abstract = rnaabstract.direct_abstraction_wrapper
