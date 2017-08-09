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









class ClusterClassifier_old():
    def __init__(self,debug=False, vectorizer=Vectorizer()):
        self.debug=debug
        self.vectorizer=vectorizer

    def fit(self, subgraphs):
        '''
        1. build NN model
        2. use this to get distances to train DBSCAN
        3. use DBSCAN clusters to train SGDClassifier
        '''
        #print "i am the clusterclassifier and i have %d subgraphs" % len(subgraphs)

        if False: # previous thing wich does something wrong
            data, subgraphs = utils.unique_graphs(subgraphs,self.vectorizer)
        else:
            subgraphs = utils.unique_graphs_graphlearn_graphhash(subgraphs)
            data= self.vectorizer.transform(subgraphs)

        #print "reduced instances to: %d \n\n" % len(subgraphs)
        # just make sure to have a backup for now
        self.data = data

        # build NN model
        NTH_NEIGHBOR = 1



        # use joachim-code:
        '''
        from bioinf_learn import MinHash
        minHash = MinHash(n_neighbors=NTH_NEIGHBOR+1)
        minHash.fit(data)
        dist, indices = minHash.kneighbors(r    eturn_distance=True)
        print dist
        neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=NTH_NEIGHBOR+1, metric='euclidean')
        neigh.fit(data)
        dist, indices = neigh.kneighbors(data)
        print dist
        '''


        # use sklearn NN
        neigh = NearestNeighbors(n_neighbors=NTH_NEIGHBOR+1, metric='euclidean')
        neigh.fit(data)
        dist, indices = neigh.kneighbors(data)
        #print dist

        # get the median
        dist = np.median(dist[:, NTH_NEIGHBOR], axis=0)
        #print dist

        #dist = 1.09

        # build DBSCAN
        scan = DBSCAN(eps=dist, min_samples=2)
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

            # ok now we want to print the INFO from above
            for cid in set(self.cluster_ids):
                print "cluster: %d  len %d" % (cid, len(self.graphclusters[cid]))
                draw.graphlearn(self.graphclusters[cid][:5])




        # deletelist = [i for i, e in enumerate(cluster_ids) if e in self.ignore_clusters]
        # targetlist = [e for e in cluster_ids if e not in self.ignore_clusters ]
        # data = delete_rows_csr(data, deletelist)
        # print targetlist
        self.cluster_classifier = SGDClassifier()
        self.cluster_classifier.fit(data, self.cluster_ids)


    def predict(self, matrix):
        return self.cluster_classifier.predict(matrix)