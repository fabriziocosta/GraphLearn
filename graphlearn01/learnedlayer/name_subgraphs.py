import eden
import numpy as np
import sklearn
import logging
logger = logging.getLogger(__name__)
from sklearn.linear_model import SGDClassifier
from graphlearn01 import utils
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from eden.graph import Vectorizer

from graphlearn01.utils import draw






class ClusterClassifier():                                                 #!!!!!!!!!!!!!

    def __init__(self,debug=False, vectorizer=Vectorizer()):
        self.debug=debug
        self.vectorizer=vectorizer

    def cluster_subgraphs(self, matrix, nth_neighbor=1):
        '''
        get the median distance to the NTH neighbor with NN
        use that distance to cluster with scan
        '''
        neigh = NearestNeighbors(n_neighbors=nth_neighbor+1, metric='euclidean')
        neigh.fit(matrix)
        dist, indices = neigh.kneighbors(matrix)
        dist = np.median(dist[:, nth_neighbor], axis=0) # 1 is the Nth neigh

        # get the clusters
        scan = DBSCAN(eps=dist, min_samples=2)
        return scan.fit_predict(matrix)

    def predict(self, matrix):
        return self.cluster_classifier.predict(matrix)

    def fit(self, subgraphs):
        # delete duplicates
        subgraphs = utils.unique_graphs_graphlearn_graphhash(subgraphs)
        matrix = self.vectorizer.transform(subgraphs)
        cluster_ids = self.cluster_subgraphs(matrix)

        self.cluster_classifier = SGDClassifier()
        self.cluster_classifier.fit(matrix, cluster_ids)



        if self.debug:
            graphclusters = defaultdict(list)
            for d,g in zip(matrix,subgraphs):
                g.graph['hash_title']= utils.hash_eden_vector(d)
            for i, cluster_id in enumerate(cluster_ids):
                # if cluster_id not in self.ignore_clusters:
                graphclusters[cluster_id].append(subgraphs[i])
            # info
            logger.debug('num clusters: %d' % max(cluster_ids))
            logger.debug(eden.util.report_base_statistics(cluster_ids).replace('\t', '\n'))

            # ok now we want to print the INFO from above
            for cid in set(cluster_ids):
                print "cluster: %d  len %d" % (cid, len(graphclusters[cid]))
                draw.graphlearn(graphclusters[cid][:5])






