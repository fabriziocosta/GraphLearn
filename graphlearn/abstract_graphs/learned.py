from eden.modifier.graph.structure import contraction
from sklearn.cluster import KMeans
from graphlearn.abstract_graphs.abstract import AbstractWrapper
from graphlearn.estimator import Wrapper as estimartorwrapper
from graphlearn.processing import PreProcessor
from graphlearn.utils import draw
import eden

'''
file contains:
    a preprocessor that takes care of abstraction

the idea here is to learn how to create the graph minor.

the preprocessor usualy gets all the input graphs in the beginning.
with those it tries to find out how to make a graph minor.
the peprocessor object will then be used to create minors for all the graphs that
appear during sampling.
'''

class PreProcessor(PreProcessor):

    def __init__(self,base_thickness_list=[2],kmeans_clusters=4):
        '''
        Args:
            base_thickness_list: [int]
                graphlearn takes care of thickness and radius, but it doesnt know about our base thickness ooO
            kmeans_clusters: int
                how many clusters to make
        Returns:
        '''
        self.base_thickness_list= base_thickness_list
        self.kmeans_clusters=kmeans_clusters

    def fit(self,inputs):
        '''

        Args:
            inputs: [graph] (no wrappers)
                something to train the estimator and do clustering
        Returns: void
            nothing
        '''
        self.rawgraph_estimator= estimartorwrapper(nu=.3, n_jobs=4)
        self.rawgraph_estimator.fit(inputs, vectorizer=self.vectorizer)
        self.make_kmeans(inputs)


    def make_kmeans(self, inputs):
        """

        Args:
            inputs: [graph]

        Returns:
            will fit self.kmeans
        """
        li=[]
        for graph in inputs:
            g=self.vectorizer.annotate([graph], estimator=self.rawgraph_estimator.estimator).next()
            for n,d in g.nodes(data=True):
                li.append([d['importance']])


        self.kmeans = KMeans(n_clusters=self.kmeans_clusters)
        self.kmeans.fit(li)


    def fit_transform(self,inputs):
        '''
        Parameters
        ----------
        input : graphs

        Returns
        -------
            graphwrapper list
        '''
        inputs=list(inputs)
        self.fit(inputs)
        return self.transform(inputs)

    def re_transform_single(self, graph):
        '''

        Args:
            graph: a raw graph

        Returns:
            wrapped graph
        '''
        #draw.graphlearn(graph)
        #print len(graph)
        abstract=self.abstract(graph,debug=False)
        #draw.graphlearn([graph,abstract])
        return AbstractWrapper(graph,vectorizer=self.vectorizer,base_thickness_list=self.base_thickness_list,abstract_graph=abstract)



    def abstract(self,graph, score_attribute='importance', group='class', debug=False):
        '''
        Args:
            graph: raw graph
            score_attribute: string
                in which attribute do i write the score
            group: string
                in which attribute do i write the cluster id that was assigned
            debug: bool
                nothing to see here
        Returns:
            abstract graph that was created from 'graph'
        '''

        graph = self.vectorizer._edge_to_vertex_transform(graph)
        graph2 = self.vectorizer._revert_edge_to_vertex_transform(graph)

        if debug:
            print 'abstr here1'
            draw.graphlearn(graph2)

        graph2 = self.vectorizer.annotate([graph2], estimator=self.rawgraph_estimator.estimator).next()


        for n,d in graph2.nodes(data=True):
            d[group]=str(self.kmeans.predict(d[score_attribute])[0])


        if debug:
            print 'abstr here'
            draw.graphlearn(graph2, vertex_label='class')



        graph2 = contraction([graph2], contraction_attribute=group, modifiers=[], nesting=False).next()


        ''' THIS LISTS ALL THE LABELS AND HASHES THEM
        for n,d in graph2.nodes(data=True):
            names=[]
            for node in d['contracted']:
                names.append(graph.node[node]['label'])
            names.sort()
            names=''.join(names)
            d['label']=str(hash(names))
        '''


        graph2 = self.vectorizer._edge_to_vertex_transform(graph2)

        #  is this mainly for coloring?
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

        Args:
            inputs: [graph]

        Returns: [graphwrapper]

        '''
        return [ AbstractWrapper(self.vectorizer._edge_to_vertex_transform(i),
                                 vectorizer=self.vectorizer,
                                 base_thickness_list=self.base_thickness_list,
                                 abstract_graph=self.abstract(i)) for i in inputs]

