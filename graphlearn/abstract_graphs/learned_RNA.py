import RNA as rna
from graphlearn.abstract_graphs.abstract import AbstractWrapper
from graphlearn.processing import PreProcessor
from graphlearn.utils import draw


class RnaPreProcessor(PreProcessor):

    def __init__(self,base_thickness_list=[2], kmeans_clusters=2):
        self.base_thickness_list= base_thickness_list
        self.kmeans_clusters=kmeans_clusters

    def fit(self, inputs):
        self.NNmodel=rna.NearestNeighborFolding()
        self.NNmodel.fit(inputs,4)
        abstr_input = [self._sequence_to_base_graph(seq) for seq in inputs ]
        self.make_abstract = PreProcessor(self.base_thickness_list,self.kmeans_clusters)
        self.make_abstract.fit(abstr_input)

        return self

    def fit_transform(self,inputs):
        '''
        Parameters
        ----------
        input : many inputs

        Returns
        -------
        graphwrapper iterator
        '''
        inputs=list(inputs)

        self.fit(inputs,self.vectorizer)
        return self.transform(inputs)

    def re_transform_single(self, graph):
        '''

        Parameters
        ----------
        graphwrapper

        Returns
        -------
        a postprocessed graphwrapper
        '''
        try:
            sequence = rna.get_sequence(graph)
        except:
            from graphlearn.utils import draw
            print 'sequenceproblem:'
            draw.graphlearn(graph, size=20)
            return None

        sequence= sequence.replace("F",'')
        return self.transform([sequence])[0]



    def _sequence_to_base_graph(self,sequence):
        structure = self.NNmodel.transform_single(sequence)
        structure,sequence= rna.fix_structure(structure,sequence)
        base_graph = rna.converter.sequence_dotbracket_to_graph(seq_info=sequence, \
                                                                seq_struct=structure)
        return base_graph


    def transform(self,sequences):
        """

        Parameters
        ----------
        sequences : iterable over rna sequences

        Returns
        -------
        list of RnaGraphWrappers
        """
        result=[]
        for sequence in sequences:
            if type(sequence)==str:
                structure = self.NNmodel.transform_single(sequence)
                structure,sequence= rna.fix_structure(structure,sequence)
                base_graph = rna.converter.sequence_dotbracket_to_graph(seq_info=sequence, \
                                                                        seq_struct=structure)
                abstract_graph=self.make_abstract.abstract(base_graph.copy())

                base_graph = self.vectorizer._edge_to_vertex_transform(base_graph)
                base_graph = rna.expanded_rna_graph_to_digraph(base_graph)

                result.append(rna.RnaWrapper(sequence, structure,base_graph, self.vectorizer, self.base_thickness_list,\
                                             abstract_graph=abstract_graph))



            # up: normal preprocessing case, down: hack to avoid overwriting the postprocessor
            else:
                result.append(self.re_transform_single(sequence))
        return result


"""
below: mole version
"""


class ScoreGraphWrapper(AbstractWrapper):
    def abstract_graph(self):
        return self._abstract_graph

    #def __init__(self,graph,vectorizer=eden.graph.Vectorizer(), base_thickness_list=None):
    def __init__(self, abstr,graph,vectorizer=None, base_thickness_list=None):
        self.some_thickness_list=base_thickness_list
        self.vectorizer=vectorizer
        self._base_graph=graph
        self._abstract_graph=abstr
        self._mod_dict={}