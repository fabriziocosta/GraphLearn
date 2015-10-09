import math
from eden.modifier.graph.structure import contraction
from graphlearn.estimator import Wrapper as estimartorwrapper
from graphlearn.utils import draw
from sklearn.cluster import KMeans
'''
  probably we wont overwrite this.. .,., delete if forgotten
from graphlearn.graphlearn import GraphLearnSampler
class Sampler(GraphLearnSampler):

    def fit(self, input, n_jobs=-1, nu=.5, batch_size=10):
        """
          use input to fit the grammar and fit the estimator
        """

        graphmanagers = self.preprocessor.fit_transform(input, self.vectorizer)

        self.estimatorobject.fit(graphmanagers,
                                                  vectorizer=self.vectorizer,
                                                  nu=nu,
                                                  n_jobs=n_jobs,
                                                  random_state=self.random_state)
        self.lsgg.fit(graphmanagers, n_jobs, batch_size=batch_size)
'''





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
        print len(graph)
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