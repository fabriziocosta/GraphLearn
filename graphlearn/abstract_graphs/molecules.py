import graphlearn.abstract_graphs.rnaabstract
from ubergraphlearn import UberSampler,UberGrammar
import ubergraphlearn
import networkx as nx
import graphlearn.utils.draw as draw



class MoleculeSampler(UberSampler):



    def _sample_init(self, graph):
        self.postprocessor.fit(self)
        graph=self.postprocessor.postprocess(graph)
        self._score(graph)
        self._sample_notes = ''
        self._sample_path_score_set = set()
        return graph

    def _score(self,graph):
        estimateable=graph.graphmanager.get_estimateable()
        super(MoleculeSampler,self)._score(estimateable)
        graph._score=estimateable._score
        return graph._score



class GraphManager(object):
    '''
    these are the basis for creating a fitting an ubersampler
    def get_estimateable(self):
    def get_base_graph(self):
    def get_abstract_graph(self):

    '''
    def __init__(self,sequence_name,sequence,vectorizer,structure):
        self.sequence_name=sequence_name
        self.sequence=sequence
        self.vectorizer=vectorizer
        self.structure=structure

        # create a base_graph , expand
        base_graph=conv.sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=structure)
        self.base_graph=vectorizer._edge_to_vertex_transform(base_graph)

        # get an expanded abstract graph
        abstract_graph=forgi.get_abstr_graph(structure)
        abstract_graph=vectorizer._edge_to_vertex_transform(abstract_graph)

        #connect edges to nodes in the abstract graph
        self.abstract_graph=edge_parent_finder(abstract_graph,self.base_graph)


        # we are forced to set a label .. for eden reasons
        def name_edges(graph,what=''):
            for n,d in graph.nodes(data=True):
                if 'edge' in d:
                    d['label']=what

        # in the abstract graph , all the edge nodes need to have a contracted attribute.
        # originaly this happens naturally but since we make multiloops into one loop there are some left out
        def setset(graph):
            for n,d in graph.nodes(data=True):
                if 'contracted' not in d:
                    d['contracted']=set()

        name_edges(self.abstract_graph)
        setset(self.abstract_graph)


    def get_estimateable(self):

        # returns an expanded, undirected graph
        # that the eden machine learning can compute
        return nx.disjoint_union(self.base_graph,self.abstract_graph)

    def get_base_graph(self):
        if 'directed_base_graph' not in self.__dict__:
            self.directed_base_graph= graphlearn.abstract_graphs.rnaabstract.expanded_rna_graph_to_digraph(self.base_graph)

        return self.directed_base_graph

    def get_abstract_graph(self):
        return self.abstract_graph
