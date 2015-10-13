
from eden.modifier.graph.structure import contraction
from graphlearn.estimator import Wrapper as estimartorwrapper
from graphlearn.utils import draw
from sklearn.cluster import KMeans
import RNA as rna




class RnaPreProcessor(object):

    def __init__(self,base_thickness_list=[2], kmeans_clusters=2):
        self.base_thickness_list= base_thickness_list
        self.kmeans_clusters=kmeans_clusters

    def fit(self, inputs,vectorizer):
        self.vectorizer=vectorizer
        self.NNmodel=rna.NearestNeighborFolding()
        self.NNmodel.fit(inputs,4)
        abstr_input = [self._sequence_to_base_graph(seq) for seq in inputs ]
        self.make_abstract = PreProcessor(self.base_thickness_list,self.kmeans_clusters)
        self.make_abstract.fit(abstr_input,vectorizer)

        return self

    def fit_transform(self,inputs,vectorizer):
        '''
        Parameters
        ----------
        input : many inputs

        Returns
        -------
        graphwrapper iterator
        '''
        inputs=list(inputs)

        self.fit(inputs,vectorizer)
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

class PreProcessor(object):

    def __init__(self,base_thickness_list=[2],kmeans_clusters=4):
        self.base_thickness_list= base_thickness_list
        self.kmeans_clusters=kmeans_clusters

    def fit(self,inputs,vectorizer):
        self.vectorizer=vectorizer
        self.raw_estimator= estimartorwrapper()
        self.raw_estimator.fit(inputs,vectorizer=self.vectorizer, nu=.3, n_jobs=4 )
        self.make_kmeans(inputs)


    def make_kmeans(self, inputs):
        li=[]
        for graph in inputs:
            g=self.vectorizer.annotate([graph], estimator=self.raw_estimator.estimator).next()
            for n,d in g.nodes(data=True):
                li.append([d['importance']])


        self.kmeans = KMeans(n_clusters=self.kmeans_clusters)
        self.kmeans.fit(li)


    def fit_transform(self,inputs,vectorizer):
        '''
        Parameters
        ----------
        input : many inputs

        Returns
        -------
        graphwrapper iterator
        '''

        inputs=list(inputs)
        self.fit(inputs,vectorizer)
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

        draw.graphlearn(graph)
        #print len(graph)
        abstract=self.abstract(graph,debug=False)
        draw.graphlearn([graph,abstract])
        return ScoreGraphWrapper(abstract,graph,self.vectorizer,self.base_thickness_list)


    def abstract(self,graph, score_attribute='importance', group='class', debug=False):
        '''
        Parameters
        ----------
        score_attribute
        group

        Returns
        -------
        '''

        graph = self.vectorizer._edge_to_vertex_transform(graph)
        graph2 = self.vectorizer._revert_edge_to_vertex_transform(graph)

        if debug:
            print 'abstr here1'
            draw.graphlearn(graph2)

        graph2 = self.vectorizer.annotate([graph2], estimator=self.raw_estimator.estimator).next()

        for n,d in graph2.nodes(data=True):
            #d[group]=str(math.floor(d[score_attribute]))
            d[group]=str(self.kmeans.predict(d[score_attribute])[0])

        if debug:
            print 'abstr here'
            draw.graphlearn(graph2, vertex_label='class')

        graph2 = contraction([graph2], contraction_attribute=group, modifiers=[], nesting=False).next()
        graph2 = self.vectorizer._edge_to_vertex_transform(graph2)

        # find out to which abstract node the edges belong
        # finding out where the edge-nodes belong, because the contractor cant possibly do this
        getabstr = {contra: node for node, d in graph2.nodes(data=True) for contra in d.get('contracted', [])}

        for n, d in graph.nodes(data=True):
            if 'edge' in d:
                # if we have found an edge node...
                # lets see whos left and right of it:
                n1, n2 = graph.neighbors(n)
                # case1: ok those belong to the same gang so we most likely also belong there.
                if getabstr[n1] == getabstr[n2]:
                    graph2.node[getabstr[n1]]['contracted'].add(n)

                # case2: neighbors belong to different gangs...
                else:
                    blub = set(graph2.neighbors(getabstr[n1])) & set(graph2.neighbors(getabstr[n2]))
                    for blob in blub:
                        if 'contracted' in graph2.node[blob]:
                            graph2.node[blob]['contracted'].add(n)
                        else:
                            graph2.node[blob]['contracted'] = set([n])
        return graph2


    def transform(self,inputs):
        '''

        Parameters
        ----------
        inputs : list of things

        Returns
        -------
        graphwrapper : iterator
        '''
        return [ ScoreGraphWrapper(self.abstract(i),self.vectorizer._edge_to_vertex_transform(i),self.vectorizer,self.base_thickness_list) for i in inputs]



from graphlearn.abstract_graphs.ubergraphlearn import UberWrapper
class ScoreGraphWrapper(UberWrapper):
    def abstract_graph(self):
        return self._abstract_graph

    #def __init__(self,graph,vectorizer=eden.graph.Vectorizer(), base_thickness_list=None):
    def __init__(self, abstr,graph,vectorizer=None, base_thickness_list=None):
        self.some_thickness_list=base_thickness_list
        self.vectorizer=vectorizer
        self._base_graph=graph
        self._abstract_graph=abstr
        self._mod_dict={}