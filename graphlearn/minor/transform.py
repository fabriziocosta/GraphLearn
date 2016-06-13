"""
here we learn how to generate a graph minor such that
later transform:graph->(graph,graphminor)
"""
from eden.modifier.graph.structure import contraction
from collections import defaultdict
from graphlearn.estimate import OneClassEstimator
from graphlearn.transform import GraphTransformer
import  graphlearn.utils.draw as draw
import networkx as nx
import logging
from itertools import izip
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from eden.util import report_base_statistics
logger = logging.getLogger(__name__)
from eden.graph import Vectorizer


class GraphToAbstractTransformer(object):
    '''
    makes abstractions that are based on the score of an estimator

    this class is just a helper for minor transform.
    '''


    def __init__(self, estimator=False,grouper=False, score_threshold=0, min_size=0, debug=False):
        '''

        Parameters
        ----------
        vectorizer  eden.graph.vectorizer
        estimator   estimator to assign scores
        grouper
            object with predict(score) function to assign clusterid to nodes
        score_threshold
            ignore nodes with score < thresh
        min_size
            min size for clusters
        debug
            debug mode?

        Returns
        -------

        '''
        self.vectorizer = Vectorizer()
        self.grouper=grouper
        self.estimator=estimator
        self.score_threshold=score_threshold
        self.min_size=min_size
        self.debug=debug
    """
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
    """
    def _transform_single(self, graph, score_attribute='importance', group='class'):
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
        graph_unexp = self.vectorizer._revert_edge_to_vertex_transform(graph_exp)


        # annotate with scores, then transform scores to clusterid
        graph_unexp = self.vectorizer.annotate([graph_unexp], estimator=self.estimator.estimator).next()
        for n, d in graph_unexp.nodes(data=True):
            if d[score_attribute] > self.score_threshold:
                d[group] = str(self.grouper.predict(d[score_attribute])[0])
            else:d[group]="-"

        if self.debug:
            print 'graph2: after score annotation, "-"-> value below thresh'
            draw.graphlearn(graph_unexp, vertex_label=group)

        # weed out groups that are too small
        # assign_values_to_nodelabel(graph_unexp, group)
        graph3 = contraction([graph_unexp], contraction_attribute=group, modifiers=[], nesting=False).next()
        for n,d in graph3.nodes(data=True):
            if len(d['contracted']) < self.min_size:
                for n in d['contracted']:
                    graph_unexp.node[n].pop(group)
                    graph_unexp.node[n][group]='-'

        #assign_values_to_nodelabel(graph_unexp, group)
        if self.debug:
            print 'weed out more nodes because the clusters are too small'
            print '[contraction, actually interesting thing]'
            draw.graphlearn([graph3,graph_unexp], vertex_label=group)
        graph_unexp = contraction([graph_unexp], contraction_attribute=group, modifiers=[], nesting=False).next()
        for n,d in graph_unexp.nodes(data=True):
            if d[group]=='-':
                d['APPROVEDABSTRACTNODE'] = False
        if self.debug:
            print 'contracts to this:'
            draw.graphlearn(graph_unexp, vertex_label=group)
        # expand
        graph_reexp = self.vectorizer._edge_to_vertex_transform(graph_unexp)
        #  make a dictionary that maps from base_graph_node -> node in contracted graph
        getabstr = {contra: node for node, d in graph_reexp.nodes(data=True) for contra in d.get('contracted', [])}

        # so this basically assigns edges in the base_graph to nodes in the abstract graph.
        for n, d in graph_exp.nodes(data=True):
            if 'edge' in d:
                # if we have found an edge node...
                # lets see whos left and right of it:
                n1, n2 = graph_exp.neighbors(n)
                # case1: ok those belong to the same gang so we most likely also belong there.
                if getabstr[n1] == getabstr[n2]:
                    graph_reexp.node[getabstr[n1]]['contracted'].add(n)

                # case2: neighbors belong to different gangs...
                else:
                    blub = set(graph_reexp.neighbors(getabstr[n1])) & set(graph_reexp.neighbors(getabstr[n2]))
                    for blob in blub:
                        if 'contracted' in graph_reexp.node[blob]:
                            graph_reexp.node[blob]['contracted'].add(n)
                        else:
                            graph_reexp.node[blob]['contracted'] = set([n])

        return graph_reexp

    def transform(self, graphs, score_attribute='importance', group='class', debug=False):
        for graph in graphs:
            yield self._transform_single(graph, score_attribute=score_attribute,group=group,debug=debug)






class GraphMinorTransformer(GraphTransformer):
    def __init__(self,
                 core_shape_cluster=KMeans(n_clusters=4),
                 name_cluster=MiniBatchKMeans(n_clusters=5),
                 save_graphclusters=False,
                 #graph_to_minor=GraphToAbstractTransformer(),
                 estimator=OneClassEstimator(nu=.5, n_jobs=4),
                 shape_min_size=1,
                 shape_score_threshold=0):
        '''

        Parameters
        ----------
        core_shape_cluster: KMeans()
            fittable cluster algo that clusters estimator scores
            you may also just use raw scores which works best with the shape_* parameters
        name_cluster: MiniBatchKMeans()
            fitable cluster algo that will run on core_shape_clusters

        save_graphclusters:
            saving the extracted core_shape_clusters
            ans order them by their name_cluster
        estimator
            oneclass estimator that will work on vectorized whole graphs

        shape_min_size:
            influencing how a core(minor node) may look like, here we set a minimum size for the code
        shape_score_threshold:
            influencing how a core(minor node) may look like,
            here we set a minimum score for the core.

        Returns
        -------

        '''
        self.save_graphclusters = save_graphclusters
        self.name_cluster = name_cluster
        self.core_shape_cluster = core_shape_cluster
        self.rawgraph_estimator = estimator
        self.shape_score_threshold=shape_score_threshold
        self.shape_min_size=shape_min_size
        self.vectorizer = Vectorizer()


    def fit(self, inputs):
        '''

        Parameters
        ----------
        inputs many nx.graph

        Returns
        -------
        '''
        # this k means is over the values resulting from annotation
        # and determine how a graph will be split intro minor nodes.
        vectorized_inputs= self.vectorizer.transform(inputs)
        self.rawgraph_estimator.fit(vectorized_inputs)


        self.train_core_shape_cluster(inputs)

        self._abstract=GraphToAbstractTransformer(score_threshold=self.shape_score_threshold,
                                                  min_size=self.shape_min_size,
                                                  debug=False,
                                                  estimator=self.rawgraph_estimator,
                                                  grouper=self.core_shape_cluster)


        # now comes the second part in which i try to find a name for those minor nodes.
        if self.name_cluster:
            parts = []
            # for all minor nodes:
            for graph in inputs:
                #       self._abstract._transform_single(graph, score_attribute, group)
                abstr = self._abstract._transform_single(graph, score_attribute='importance', group='class')
                for n, d in abstr.nodes(data=True):
                    if len(d['contracted']) > 1 and 'edge' not in d and d.get('APPROVEDABSTRACTNODE',True):
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




    def train_core_shape_cluster(self, inputs):
        '''

        Parameters
        ----------
        inputs: [graph]

        Returns
        -------

        '''
        li = []
        for graph in inputs:
            g = self.vectorizer.annotate([graph], estimator=self.rawgraph_estimator.estimator).next()
            for n, d in g.nodes(data=True):
                li.append([d['importance']])

        self.core_shape_cluster.fit(li)




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
        #abstract = self.abstract(graph, debug=False)
        # draw.graphlearn([graph,abstract])
        return self.transform([graph])[0]


    def abstract(self, graph, score_attribute='importance', group='class', debug=False):
        '''
        Parameters
        ----------
        graph: nx.graph
        score_attribute: string, 'importance'
            attribute in which we write the score
        group: string, 'class'
            where to write clusterid
        debug: bool, False
            draw abstracted graphs

        Returns
        -------
            nx.graph: a graph minor
            if named_cluster is present, we the minor graph nodes will be named accordingly
        '''


        # generate abstract graph
        abst = self._abstract._transform_single(graph, score_attribute, group)

        if self.name_cluster==False:
            return abst

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

        Parameters
        ----------
        inputs: [graph]

        Returns
        -------
            list of decomposers
        '''

        return [ (self.vectorizer._edge_to_vertex_transform(graph),self.abstract(graph)) for graph in inputs ]



