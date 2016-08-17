"""
here we learn how to generate a graph minor such that
later transform:graph->(graph,graphminor)


fit will first train a oneclass estimator on the input graphs.

then we find out which nodes will be grouped together to an abstract node.
this can be done via group_score_threshold, where high scoring nodes will be placed together.
the alternative is to look at all the estimator scores for each node and k-mean them.
with both strategies, only groups of a certain minimum size will be contracted to a minor node.

in the end all groups can be automatically clustered and named by their cluster(name_cluster).

THIS IS A TEST I AM TRYING TO MAKE THIS MORE EZ AND EZ TO TEST


"""

import abstractor
from collections import defaultdict
from graphlearn.estimate import ExperimentalOneClassEstimator
from graphlearn.transform import GraphTransformer
import graphlearn.utils.draw as draw
import networkx as nx
import logging
import sklearn
from itertools import izip
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from eden.util import report_base_statistics

logger = logging.getLogger(__name__)
from eden.graph import Vectorizer
from eden import graph as edengraphtools
from sklearn.linear_model import SGDClassifier
from GArDen.model import ClassifierWrapper
import numpy as np
import eden
import multiprocessing as mp


class myclusterclassifier():
    def __init__(self):
        pass

    def fit(self, data):
        '''
        1. build NN model
        2. use this to get distances to train DBSCAN
        3. use DBSCAN clusters to train SGDClassifier
        '''

        # just make sure to have a backup for now
        self.data = data

        # build NN model
        '''
        NTH_NEIGHBOR = 1
        # use joachim-code:
        from bioinf_learn import MinHash
        minHash = MinHash(n_neighbors=NTH_NEIGHBOR+1)
        minHash.fit(data)
        dist, indices = minHash.kneighbors(return_distance=True)
        print dist

        # use sklearn NN
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=NTH_NEIGHBOR+1, metric='euclidean')
        neigh.fit(data)
        dist, indices = neigh.kneighbors(data)
        print dist

        # get the median
        dist = np.median(dist[:, NTH_NEIGHBOR], axis=0)
        print dist
        '''
        dist = 1.09

        # build DBSCAN
        scan = sklearn.cluster.DBSCAN(eps=dist, min_samples=2)
        self.cluster_ids = scan.fit_predict(data)

        # filter clusters that are too small or too large , NOT NOW
        # '''
        # for i in range(self.subgraph_name_estimator.get_params()['n_clusters']):
        #    cids = cluster_ids.tolist()
        #    members = cids.count(i)
        #    if members< self.cluster_min_members or members >  self.cluster_max_members > -1: # should work to omou
        #        logger.debug('remove cluser: %d  members: %d' % (i,members))
        #        self.ignore_clusters.append(i)
        #
        # '''


        # info
        logger.debug('num clusters: %d' % max(self.cluster_ids))
        logger.debug(report_base_statistics(self.cluster_ids).replace('\t', '\n'))

        # deletelist = [i for i, e in enumerate(cluster_ids) if e in self.ignore_clusters]
        # targetlist = [e for e in cluster_ids if e not in self.ignore_clusters ]
        # data = delete_rows_csr(data, deletelist)
        # print targetlist
        self.cluster_classifier = SGDClassifier()
        self.cluster_classifier.fit(data, self.cluster_ids)

    def predict(self, matrix):
        return self.cluster_classifier.predict(matrix)


class GraphMinorTransformer(GraphTransformer):
    def __init__(self,
                 vectorizer=Vectorizer(complexity=3),
                 estimator=ExperimentalOneClassEstimator(),
                 group_min_size=2,
                 group_max_size=5,
                 # cluster_min_members=0,
                 # cluster_max_members=-1,
                 group_score_threshold=0.4,
                 debug=False,
                 # subgraph_cluster=,
                 cluster_classifier=myclusterclassifier(),
                 # save_graphclusters=False,
                 multiprocess=True,
                 layer=0):
        '''
        initial subgraphs are identified by threshold per default,
        you can also cluster those by setting group_score_classifier
        to eg sklearn.cluster.KMeans

        Parameters
        ----------
        vectorizer: eden vectorizer
        estimator: graphlearn estimator wrapper
        group_min_size: int
        group_max_size int
        cluster_min_members:int
        cluster_max_members:int
        group_score_threshold: float
        group_score_classifier: KMeans(n_clusters=4)
        debug: bool
        subgraph_cluster: MiniBatchKMeans
        '''
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.max_size = group_max_size
        self.min_size = group_min_size
        self.score_threshold = group_score_threshold
        self.debug = debug
        # self.subgraph_cluster=subgraph_cluster
        self.cluster_classifier = cluster_classifier
        # self.save_graphclusters=save_graphclusters
        # self.cluster_min_members=cluster_min_members
        # self.cluster_max_members=cluster_max_members
        self.layer = layer
        self.multiprocess = multiprocess

    def call_annotator(self, graphs):
        return mass_annotate_mp(graphs, self.vectorizer, score_attribute='importance',
                                estimator=self.estimator.superesti,
                                multi_process=self.multiprocess)

    def prepfit(self):
        self.abstractor = abstractor.GraphToAbstractTransformer(
            score_threshold=self.score_threshold,
            min_size=self.min_size,
            max_size=self.max_size,
            debug=self.debug,
            estimator=None,
            layer=self.layer,
            vectorizer=self.vectorizer)

    def fit(self, graphs, fit_transform=False):
        '''
        TODO: be sure to set the self.cluster_ids :)

        Parameters
        ----------
        graphs

        Returns
        -------

        '''
        #  PREPARE
        graphs = list(graphs)
        if graphs[0].graph.get('expanded', False):
            raise Exception('give me an unexpanded graph')
        # if self.layer > 0: graphs = map(lambda x:select_layer(x,self.layer),graphs)
        self.prepfit()

        # info
        if self.debug:
            print 'minortransform fit. input after select layer'
            draw.graphlearn(graphs[:3], contract=False, size=4, vertex_label='label')

        # TRAIN ESTIMATOR, GET SUBGRAPHS
        self.estimator.fit(self.vectorizer.transform(graphs))
        # self.abstractor.estimator=self.estimator.estimator
        graphs = self.call_annotator(graphs)
        subgraphs = list(self.abstractor.get_subgraphs(graphs))

        # info
        if self.debug:
            print 'minortransform fit. this is what the subgraphs i got from the abstractor look like'
            draw.graphlearn(subgraphs[:5], contract=False, size=3, edge_label='label')

        # FILTER UNIQUES AND TRAIN THE CLUSTERER
        data, subgraphs = unique_graphs(subgraphs, self.vectorizer)
        self.cluster_classifier.fit(data)
        self.abstractor.nameestimator = self.cluster_classifier

        # save the clusters because they look pretty :)
        self.graphclusters = defaultdict(list)
        for i, cluster_id in enumerate(self.cluster_classifier.cluster_ids):
            # if cluster_id not in self.ignore_clusters:
            self.graphclusters[cluster_id].append(subgraphs[i])

        # annotating is super slow. so in case of fit_transform i can save that step
        if fit_transform:
            return self.transform(graphs)

    def fit_transform(self, inputs):
        return self.fit(inputs, fit_transform=True)

    def transform(self, graphs):
        '''
        Parameters
        ----------
        inputs: [graph]

        Returns
        -------
            [(edge_expanded_graph, minor),...]
        '''

        graphs = self.call_annotator(graphs)
        result = _transform(graphs, self.layer, self.abstractor)
        if self.debug:
            print 'minortransform  transform.  1. the new layer ; 2. the old layer(s) are above :) '
            draw.graphlearn(result[:3], contract=False, size=6, vertex_label='contracted')
        return result


def _transform(graphs, layer, abstractor):
    return [re_transform_single(graph, layer, abstractor) for graph in graphs]


def re_transform_single(graph, layer, abstractor):
    '''
    Parameters
    ----------
    graph

    Returns
    -------
    a postprocessed graphwrapper
    '''
    if graph.graph.get('expanded', False):
        raise Exception('give me an unexpanded graph')
    if layer != 0:
        l_graph = select_layer(graph, layer)
    else:
        l_graph = graph
    transformed_graph = abstractor._transform_single(l_graph.copy(), apply_name_estimation=True)
    for n, d in transformed_graph.nodes(data=True):
        d['layer'] = layer + 1
    transformed_graph.graph['contracted_layers'] = layer + 1
    transformed_graph.graph['original'] = graph

    return transformed_graph


def select_layer(g, layer):
    return g.subgraph([n for n, d in g.nodes(data=True) if d.get('layer') == layer])


def unique_graphs(graphs, vectorizer):
    # returns datamatrix, subgraphs
    map(lambda x: abstractor.node_operation(x, lambda n, d: d.pop('weight', None)), graphs)
    data = vectorizer.transform(graphs)
    # remove duplicates   from data and subgraph_list
    data, indices = unique_csr(data)
    graphs = [graphs[i] for i in indices]
    return data, graphs


def delete_rows_csr(mat, indices, keep=False):
    '''
    Parameters
    ----------
    mat   csr matrix
    indices  list of indices to work on
    keep  should i delete or keep the indices

    Returns
    -------
        csr matrix
    '''
    indices = list(indices)
    if keep == False:
        mask = np.ones(mat.shape[0], dtype=bool)
    else:
        mask = np.zeros(mat.shape[0], dtype=bool)
    mask[indices] = keep
    return mat[mask]


def unique_csr(csr):
    # returns unique csr and a list of used indices
    hash_function = lambda vec: hash(tuple(vec.data + vec.indices))
    unique = {hash_function(row): ith for ith, row in enumerate(csr)}
    indices = [ith for hashvalue, ith in unique.items()]
    indices.sort()
    return delete_rows_csr(csr, indices, keep=True), indices


def mass_annotate_mp(inputs, vectorizer, score_attribute='importance', estimator=None, multi_process=False):
    '''
    graph annotation is slow. i dont want to do it twice in fit and predict :)
    '''
    #  1st check if already annotated
    if inputs[0].graph.get('mass_annotate_mp_was_here', False):
        return inputs

    if multi_process == False:
        # map(lambda x: abstractor.node_operation(x, lambda n, d: d.pop('weight', None)), inputs)
        res = list(vectorizer.annotate(inputs, estimator=estimator))
        res[0].graph['mass_annotate_mp_was_here'] = True
        return res
    else:
        pool = mp.Pool()
        mpres = [eden.apply_async(pool, mass_annotate_mp, args=(graphs, vectorizer, score_attribute, estimator)) for
                 graphs in eden.grouper(inputs, 50)]
        result = []
        for res in mpres:
            result += res.get()
        pool.close()
        pool.join()
        return result
