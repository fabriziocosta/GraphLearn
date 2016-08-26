import eden
import numpy as np
import sklearn
import logging
logger = logging.getLogger(__name__)
from sklearn.linear_model import SGDClassifier
from graphlearn import utils
from collections import defaultdict

class ClusterClassifier():
    def __init__(self,debug=False):
        self.debug=debug

    def fit(self, subgraphs):
        '''
        1. build NN model
        2. use this to get distances to train DBSCAN
        3. use DBSCAN clusters to train SGDClassifier
        '''

        data, subgraphs = utils.unique_graphs(subgraphs,eden.graph.Vectorizer())

        # just make sure to have a backup for now
        self.data = data

        # build NN model
        NTH_NEIGHBOR = 1



        # use joachim-code:
        '''
        from bioinf_learn import MinHash
        minHash = MinHash(n_neighbors=NTH_NEIGHBOR+1)
        minHash.fit(data)
        dist, indices = minHash.kneighbors(return_distance=True)
        print dist
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=NTH_NEIGHBOR+1, metric='euclidean')
        neigh.fit(data)
        dist, indices = neigh.kneighbors(data)
        print dist
        '''


        # use sklearn NN
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=NTH_NEIGHBOR+1, metric='euclidean')
        neigh.fit(data)
        dist, indices = neigh.kneighbors(data)
        #print dist

        # get the median
        dist = np.median(dist[:, NTH_NEIGHBOR], axis=0)
        #print dist

        #dist = 1.09

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
        # save the clusters because they look pretty :)

        self.graphclusters = defaultdict(list)
        if self.debug:
            for d,g in zip(data,subgraphs):
                g.graph['hash_title']= utils.hash_eden_vector(d)
            for i, cluster_id in enumerate(self.cluster_ids):
                # if cluster_id not in self.ignore_clusters:
                self.graphclusters[cluster_id].append(subgraphs[i])
            # info
            logger.debug('num clusters: %d' % max(self.cluster_ids))
            logger.debug(eden.util.report_base_statistics(self.cluster_ids).replace('\t', '\n'))

        # deletelist = [i for i, e in enumerate(cluster_ids) if e in self.ignore_clusters]
        # targetlist = [e for e in cluster_ids if e not in self.ignore_clusters ]
        # data = delete_rows_csr(data, deletelist)
        # print targetlist
        self.cluster_classifier = SGDClassifier()
        self.cluster_classifier.fit(data, self.cluster_ids)


    def predict(self, matrix):
        return self.cluster_classifier.predict(matrix)

