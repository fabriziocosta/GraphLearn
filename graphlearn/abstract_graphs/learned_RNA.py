import RNA as rna
from graphlearn.processing import PreProcessor
from graphlearn.abstract_graphs.learned_molecules import PreProcessor as default_preprocessor
from graphlearn.utils import draw
import logging
logger = logging.getLogger(__name__)

class RnaPreProcessor(PreProcessor):

    def __init__(self,base_thickness_list=[2], kmeans_clusters=2,structure_mod=True):
        self.base_thickness_list= [thickness*2 for thickness in base_thickness_list]
        self.kmeans_clusters=kmeans_clusters
        self.structure_mod=structure_mod

    def fit(self, inputs, vectorizer):
        self.vectorizer = vectorizer
        self.NNmodel = rna.EdenNNF(n_neighbors=4)
        self.NNmodel.fit(inputs)


        #abstr_input = [ self._sequence_to_base_graph(seq) for seq in inputs ]

        abstr_input = list( self.NNmodel.eden_rna_vectorizer.graphs(inputs) )
        self.make_abstract = default_preprocessor(self.base_thickness_list,self.kmeans_clusters)
        self.make_abstract.set_param(self.vectorizer)
        self.make_abstract.fit(abstr_input)
        print "fit pp done"
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
        inputs= [b for a,b in inputs]
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
            logger.debug('sequenceproblem: this is not an rna')
            #from graphlearn.utils import draw
            #print 'sequenceproblem:'
            #draw.graphlearn(graph, size=20)
            return None

        sequence= sequence.replace("F",'')
        return self.transform([sequence])[0]



    def _sequence_to_base_graph(self, sequence):

        structure = self.NNmodel.transform_single(sequence)
        if self.structure_mod:
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
                structure,energy = self.NNmodel.transform_single(('fake',sequence))
                #print structure
                if self.structure_mod:
                    structure,sequence= rna.fix_structure(structure,sequence)
                base_graph = rna.converter.sequence_dotbracket_to_graph(seq_info=sequence, \
                                                                        seq_struct=structure)

                abstract_graph=self.make_abstract.abstract(base_graph.copy())
                base_graph = self.vectorizer._edge_to_vertex_transform(base_graph)
                base_graph = rna.expanded_rna_graph_to_digraph(base_graph)

                result.append(rna.RnaWrapper(sequence, structure,base_graph, self.vectorizer, self.base_thickness_list,\
                                             abstract_graph=abstract_graph))

            # up: normal preprocessing case, down: hack to avoid overwriting the postprocessor
            # needs some changing obviously
            else:
                result.append(self.re_transform_single(sequence))
        return result


