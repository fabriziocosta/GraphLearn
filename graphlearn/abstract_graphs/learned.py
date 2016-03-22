from eden.modifier.graph.structure import contraction
from collections import defaultdict

from graphlearn.abstract_graphs.abstract import AbstractWrapper
from graphlearn.estimator import Wrapper as estimartorwrapper
from graphlearn.processing import PreProcessor
from graphlearn.utils import draw
import eden
import networkx as nx
import logging
from itertools import izip
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from eden.util import report_base_statistics

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
    def __init__(self, base_thickness_list=[2],
                 shape_cluster=KMeans(n_clusters=4),
                 name_cluster=MiniBatchKMeans(n_clusters=5),
                 save_graphclusters=False):
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

        save_graphclusters: bool, False
            saving learned_node_names clusters in self.graphclusters

        Returns
        -------
        void
        '''
        self.base_thickness_list = base_thickness_list
        self.save_graphclusters = save_graphclusters


        self.name_cluster = name_cluster
        self.shape_cluster = shape_cluster



    def fit(self, inputs):

        # this k means is over the values resulting from annotation
        # and determine how a graph will be split intro minor nodes.
        self.rawgraph_estimator = estimartorwrapper(nu=.3, n_jobs=4)
        self.rawgraph_estimator.fit(inputs, vectorizer=self.vectorizer)
        self.make_kmeans(inputs)


        self._abstract=graph_to_abstract()
        self._abstract.set_parmas(estimator=self.rawgraph_estimator, grouper=self.shape_cluster, vectorizer=self.vectorizer)

        # now comes the second part in which i try to find a name for those minor nodes.
        from graphlearn.utils import draw
        if self.name_cluster:
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



            self.name_cluster.fit(data_matrix)
            cluster_ids = self.name_cluster.predict(data_matrix)
            logger.debug('num clusters: %d' % max(cluster_ids))
            logger.debug(report_base_statistics(cluster_ids).replace('\t', '\n'))

            if self.save_graphclusters:
                self.graphclusters = defaultdict(list)
                for cluster_id, graph in izip(cluster_ids, parts):
                    self.graphclusters[cluster_id].append(graph)




    def make_kmeans(self, inputs):
        """

        Args:
            inputs: [graph]

        Returns:
            will fit self.kmeans
        """
        li = []
        for graph in inputs:
            g = self.vectorizer.annotate([graph], estimator=self.rawgraph_estimator.estimator).next()
            for n, d in g.nodes(data=True):
                li.append([d['importance']])

        self.shape_cluster.fit(li)




    def fit_transform(self, inputs):
        '''
        Parameters
        ----------
        input : graphs

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

        abst = self._abstract._transform_single(graph, score_attribute, group, debug)

        if 'clust' not in self.__dict__:
            return abst
        else:
            graph = self.vectorizer._revert_edge_to_vertex_transform(graph)
            for n, d in abst.nodes(data=True):
                if len(d['contracted']) > 1 and 'edge' not in d:
                    # get the subgraph induced by it (if it is not trivial)
                    tmpgraph = nx.Graph(graph.subgraph(d['contracted']))
                    vector = self.vectorizer.transform_single(tmpgraph)
                    d['label'] = "C_" + str(self.name_cluster.predict(vector))

                elif len(d['contracted']) == 1 and 'edge' not in d:
                    # get the subgraph induced by it (if it is not trivial)
                    d['label'] = graph.node[list(d['contracted'])[0]]['label']


                elif 'edge' not in d:
                    d['label'] = "F_should_not_happen"
            return abst




    def transform(self, inputs):
        '''

        Args:
            inputs: [graph]

        Returns: [graphwrapper]

        '''
        return [AbstractWrapper(self.vectorizer._edge_to_vertex_transform(i),
                                vectorizer=self.vectorizer, base_thickness_list=self.base_thickness_list,
                                abstract_graph=self.abstract(i)) for i in inputs]



class graph_to_abstract(object):

    def __init__(self):
        pass

    def set_parmas(self,**kwargs):
        '''

        Parameters
        ----------
        kwargs:
            vectorizer = a vectorizer to edge_vertex transform
            estimator = estimator object to assign scores to nodes
            grouper =  object with predict(score) function to assign clusterid to nodes
        Returns
        -------

        '''
        self.__dict__.update(kwargs)

    def _transform_single(self, graph, score_attribute='importance', group='class', debug=False):
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


        # graph expanded and unexpanded
        graph_exp = self.vectorizer._edge_to_vertex_transform(graph)
        graph2 = self.vectorizer._revert_edge_to_vertex_transform(graph_exp)

        if debug:
            print 'abstr here1'
            draw.graphlearn(graph2)


        # annotate with scores, then transform scores to clusterid
        graph2 = self.vectorizer.annotate([graph2], estimator=self.estimator.estimator).next()
        for n, d in graph2.nodes(data=True):
            d[group] = str(self.grouper.predict(d[score_attribute])[0])

        if debug:
            print 'abstr here'
            draw.graphlearn(graph2, vertex_label='class')

        # contract and expand
        graph2 = contraction([graph2], contraction_attribute=group, modifiers=[], nesting=False).next()
        graph2 = self.vectorizer._edge_to_vertex_transform(graph2)

        #  make a dictionary that maps from base_graph_node -> node in contracted graph
        getabstr = {contra: node for node, d in graph2.nodes(data=True) for contra in d.get('contracted', [])}

        # so this basically assigns edges in the base_graph to nodes in the abstract graph.
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

    def transform(self, graphs, score_attribute='importance', group='class', debug=False):
        for graph in graphs:
            yield self._transform_single(graph, score_attribute=score_attribute,group=group,debug=debug)