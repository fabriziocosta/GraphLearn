'''
transform: sequence -> (sequence, structure, base_graph, None)
'''
import forgi
import logging
import eden.converter.rna as converter
from graphlearn.abstract_graphs.rna import expanded_rna_graph_to_digraph, get_sequence, _pairs
from graphlearn.abstract_graphs.rna.fold import EdenNNF
from graphlearn.transform import GraphTransformer
logger = logging.getLogger(__name__)




class GraphTransformerForgi(GraphTransformer):
    def __init__(self):


        '''
        Parameters
        ----------
        base_thickness_list: list of int
            thickness list for the base graph
        structure_mod : bool
            should we introduce "F" nodes to keep multiloop flexible regarding substitution
        include_base : base
            if asked for all cips, i will also yield   "normal" cips (whose core is not radius of abstract, but radius of base graph)
        ignore_inserts:
            bolges will be ignored and merged to their adjacend stems
        Returns
        -------

        '''


    def fit(self, inputs, vectorizer):
        """

        Parameters
        ----------
        inputs: sequence list
        vectorizer: a vectorizer

        Returns
        -------
        self
        """

        self.vectorizer = vectorizer
        self.NNmodel = EdenNNF(n_neighbors=4)
        self.NNmodel.fit(inputs)
        return self

    def fit_transform(self, inputs):
        """

        Parameters
        ----------
        inputs: sequences

        Returns
        -------
        many graphdecomposers
        """

        inputs = list(inputs)
        self.fit(inputs, self.vectorizer)
        inputs = [b for a, b in inputs]
        return self.transform(inputs)

    def re_transform_single(self, graph):
        """

        Parameters
        ----------
        graph: digraph

        Returns
        -------
        graph decomposer
        """

        try:
            sequence = get_sequence(graph)
        except:
            logger.debug('sequenceproblem: this is not an rna')
            # draw.graphlearn(graph, size=20)
            return None

        sequence = sequence.replace("F", '')
        trans = self.transform([sequence])[0]
        # if trans._base_graph.graph['energy'] > -10:
        #    return None
        return trans

    def abstract_graph(self):
        '''
        we need to make an abstraction Ooo
        '''

        # create the abstract graph and populate the contracted set
        abstract_graph = forgi.get_abstr_graph(self.structure, ignore_inserts=self.ignore_inserts)
        abstract_graph = self.vectorizer._edge_to_vertex_transform(abstract_graph)
        completed_abstract_graph = forgi.edge_parent_finder(abstract_graph, self._base_graph)

        # eden is forcing us to set a label and a contracted attribute.. lets do this
        for n, d in completed_abstract_graph.nodes(data=True):
            if 'edge' in d:
                d['label'] = 'e'
        # in the abstract graph , all the edge nodes need to have a contracted attribute.
        # originaly this happens naturally but since we make multiloops into one loop
        # there are some left out
        for n, d in completed_abstract_graph.nodes(data=True):
            if 'contracted' not in d:
                d['contracted'] = set()
        return completed_abstract_graph


    def transform(self, sequences):
        """

        Parameters
        ----------
        sequences : iterable over rna sequences

        Returns
        -------
        list of RnaGraphWrappers
        """
        result = []
        for sequence in sequences:

            # if we eat a tupple, it musst be a (name, sequence) type :)  we only want a sequence
            if type(sequence) == type(()):
                logger.warning('YOUR INPUT IS A TUPPLE, GIVE ME A SEQUENCE, SINCERELY -- YOUR RNA PREPROCESSOR')

            # get structure
            structure, energy, sequence = self.NNmodel.transform_single(('fake', sequence))
            # FIXING STRUCTURE
            structure, sequence = fix_structure(structure, sequence)
            if structure == None:
                result.append(None)
                continue


            # built base_graph
            base_graph = converter.sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=structure)
            base_graph = self.vectorizer._edge_to_vertex_transform(base_graph)
            base_graph = expanded_rna_graph_to_digraph(base_graph)
            base_graph.graph['energy'] = energy
            base_graph.graph['sequence'] = sequence
            base_graph.graph['structure'] = structure
            result.append(
                   (sequence, structure, base_graph, self.abstract_graph(base_graph))
            )

        return result


'''
default method if no nearest neighbor folding class is provided
# is never used .. so we disable it
def callRNAshapes(sequence):
    cmd = 'RNAshapes %s' % sequence
    out = sp.check_output(cmd, shell=True)
    s = out.strip().split('\n')

    for li in s[2:]:
        # print li.split()
        energy, shape, abstr = li.split()
        # if abstr == '[[][][]]':
        return shape
'''

def fix_structure(stru, stri):
    '''
    structure mod is for forgi transformation..
    in forgi, core nodes dont have to be adjacent ->  dont know why currently...
    anyway we fix this by introducing nodes with an F label.

    the problem is to check every (( and )) .
    if the bonding partners are not next to each other we know that we need to act.
    '''
    p = _pairs(stru)
    lastchar = "."
    problems = []
    for i, c in enumerate(stru):
        # checking for )) and ((
        if c == lastchar and c != '.':
            if abs(p[i] - p[i - 1]) != 1:  # the partners are not next to each other
                problems.append(i)
        # )( provlem
        elif c == '(':
            if lastchar == ')':
                problems.append(i)
        lastchar = c
    problems.sort(reverse=True)
    for i in problems:
        stru = stru[:i] + '.' + stru[i:]
        stri = stri[:i] + 'F' + stri[i:]

    return stru, stri