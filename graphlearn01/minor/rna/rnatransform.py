'''
we build the transformer for the learned rna abstraction here.
also (a tiny bit) a decomposer...

'''





# ok decomp first
from graphlearn01.minor.decompose import MinorDecomposer
import graphlearn01.minor.rna as rna
class learnedRnaDedomposer(MinorDecomposer):
    def out(self):
        if self.output_sequence:
             sequence = rna.get_sequence(self.base_graph())
             return ('',sequence.replace("F",""))

        return self.base_graph()


# now we need a transformer that combines the edenNN with the learned layer stuff

from  graphlearn01.learnedlayer.transform import GraphMinorTransformer as learntransformer
from graphlearn01.minor.rna.fold import EdenNNF
from eden.graph import _edge_to_vertex_transform
import eden_rna


class learnedRnaTransformer(learntransformer):

    def fit(self,sequences):
        self.NNmodel = EdenNNF(n_neighbors=4)
        self.NNmodel.fit(sequences)
        seslist = self.NNmodel.transform(sequences)
        return super(self.__class__, self).fit( map(ses_to_graph,seslist))


    def transform(self, sequences):

        def rebuild(graphs):
            sequences = map(rna.get_sequence,graphs)
            seslist=self.NNmodel.transform(sequences)
            return map(ses_to_graph,seslist)
        return super(self.__class__, self).transform( rebuild(sequences) )




def ses_to_graph(ses):
    structure, energy, sequence = ses
    base_graph = eden_rna.sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=structure)
    base_graph = _edge_to_vertex_transform(base_graph)
    base_graph = rna.expanded_rna_graph_to_digraph(base_graph)
    base_graph.graph['energy'] = energy
    base_graph.graph['sequence'] = sequence
    base_graph.graph['structure'] = structure
    return base_graph








