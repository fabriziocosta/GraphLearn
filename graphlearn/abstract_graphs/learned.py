from eden.modifier.graph.structure import contraction
from sklearn.cluster import KMeans
from graphlearn.abstract_graphs.abstract import AbstractWrapper
from graphlearn.estimator import Wrapper as estimartorwrapper
from graphlearn.processing import PreProcessor
from graphlearn.utils import draw
import eden
import networkx as nx
import logging

logger = logging.getLogger(__name__)
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
    def __init__(self, base_thickness_list=[2], kmeans_clusters=4, learned_node_names_clusters=0):
        '''

        Parameters
        ----------
        base_thickness_list: list of int, [2]
            thickness for the base graph
            thickness and radius for the minor are provided to the graphlearn.sampler

        kmeans_clusters: int, 4
            split nodes of the graphs into this many groups (by rating provided by an estimator)

        learned_node_names: bool False
            nodes that that belong to the same group (see above) and are adjacent
            form a minor node, if this option is enabled, we try to learn a name for
            this combined node.

        Returns
        -------
        void
        '''
        self.base_thickness_list = base_thickness_list
        self.kmeans_clusters = kmeans_clusters
        if learned_node_names_clusters > 1:
            self.learned_node_names = True
            self.learned_node_names_clusters = learned_node_names_clusters

    def fit(self, inputs):

        # this k means is over the values resulting from annotation
        # and determine how a graph will be split intro minor nodes.
        self.rawgraph_estimator = estimartorwrapper(nu=.3, n_jobs=4)
        self.rawgraph_estimator.fit(inputs, vectorizer=self.vectorizer)
        self.make_kmeans(inputs)

        # now comes the second part in which i try to find a name for those minor nodes.
        from graphlearn.utils import draw
        if self.learned_node_names:
            print 'start learned stuff'
            parts = []
            # for all minor nodes:
            for graph in inputs:
                abstr = self.abstract(graph, score_attribute='importance', group='class', debug=False)
                for n, d in abstr.nodes(data=True):
                    if len(d['contracted']) > 1 and 'edge' not in d:
                        # get the subgraph induced by it (if it is not trivial)
                        tmpgraph = nx.Graph(graph.subgraph(d['contracted']))
                        parts.append(tmpgraph)

            logger.debug("learning abstraction: %d partial graphs found" % len(parts))
            # draw.graphlearn(parts[:5], contract=False)
            # code from annotation-components.ipynb:
            data_matrix = self.vectorizer.transform(parts)

            from sklearn.cluster import MiniBatchKMeans
            self.clust = MiniBatchKMeans(n_clusters=self.learned_node_names_clusters)
            self.clust.fit(data_matrix)
            cluster_ids = self.clust.predict(data_matrix)

            logger.debug('num clusters: %d' % max(cluster_ids))
            from eden.util import report_base_statistics
            logger.debug(report_base_statistics(cluster_ids).replace('\t', '\n'))

    def make_kmeans(self, inputs):
        li = []
        for graph in inputs:
            g = self.vectorizer.annotate([graph], estimator=self.rawgraph_estimator.estimator).next()
            for n, d in g.nodes(data=True):
                li.append([d['importance']])

        self.kmeans = KMeans(n_clusters=self.kmeans_clusters)
        self.kmeans.fit(li)

    def fit_transform(self, inputs):
        '''
        Parameters
        ----------
        input : many inputs

        Returns
        -------
        graphwrapper iterator
        '''

        inputs = list(inputs)
        self.fit(inputs)
        return self.transform(inputs)

    def re_transform_single(self, graph):
        '''
        Parameters
        ----------
        graph

        Returns
        -------
        a postprocessed graphwrapper
        '''

        # draw.graphlearn(graph)
        # print len(graph)
        abstract = self.abstract(graph, debug=False)
        # draw.graphlearn([graph,abstract])
        return AbstractWrapper(graph, vectorizer=self.vectorizer, base_thickness_list=self.base_thickness_list,
                               abstract_graph=abstract)

    def abstract(self, graph, score_attribute='importance', group='class', debug=False):
        abst = self._abstract(graph, score_attribute, group, debug)
        if 'clust' not in self.__dict__:
            return abst
        else:
            graph = self.vectorizer._revert_edge_to_vertex_transform(graph)
            for n, d in abst.nodes(data=True):
                if len(d['contracted']) > 1 and 'edge' not in d:
                    # get the subgraph induced by it (if it is not trivial)
                    tmpgraph = nx.Graph(graph.subgraph(d['contracted']))
                    vector = self.vectorizer.transform_single(tmpgraph)
                    d['label'] = "C_" + str(self.clust.predict(vector))

                elif len(d['contracted']) == 1 and 'edge' not in d:
                    # get the subgraph induced by it (if it is not trivial)
                    d['label'] = graph.node[list(d['contracted'])[0]]['label']


                elif 'edge' not in d:
                    d['label'] = "F_should_not_happen"
            return abst

    def _abstract(self, graph, score_attribute='importance', group='class', debug=False):
        '''
        Parameters
        ----------
        score_attribute: string
            name of the attribute used
        group: string
            annnotate in this field
        Returns
        -------
        '''

        graph_exp = self.vectorizer._edge_to_vertex_transform(graph)
        graph2 = self.vectorizer._revert_edge_to_vertex_transform(graph_exp)

        if debug:
            print 'abstr here1'
            draw.graphlearn(graph2)

        graph2 = self.vectorizer.annotate([graph2], estimator=self.rawgraph_estimator.estimator).next()

        for n, d in graph2.nodes(data=True):
            d[group] = str(self.kmeans.predict(d[score_attribute])[0])

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
        for n, d in graph_exp.nodes(data=True):
            if 'edge' in d:
                # if we have found an edge node...
                # lets see whos left and right of it:
                n1, n2 = graph_exp.neighbors(n)
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

    def transform(self, inputs):
        '''

        Parameters
        ----------
        inputs : list of things

        Returns
        -------
        graphwrapper : iterator
        '''
        return [AbstractWrapper(self.vectorizer._edge_to_vertex_transform(i),
                                vectorizer=self.vectorizer, base_thickness_list=self.base_thickness_list,
                                abstract_graph=self.abstract(i)) for i in inputs]
