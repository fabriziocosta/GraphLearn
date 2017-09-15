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

    def __init__(self,debug=False, vectorizer=Vectorizer(),min_clustersize=2,dbscan_range=.6):
        self.debug=debug
        self.dbscan_range=dbscan_range
        self.vectorizer=vectorizer
        self.min_clustersize=min_clustersize

    def cluster_subgraphs(self, matrix, nth_neighbor=1):
        '''
        get the median distance to the NTH neighbor with NN
        use that distance to cluster with scan
        '''
        neigh = NearestNeighbors(n_neighbors=nth_neighbor+1, metric='euclidean')
        neigh.fit(matrix)
        dist, indices = neigh.kneighbors(matrix)
        #dist = np.median(dist[:, nth_neighbor], axis=0) # 1 is the Nth neigh


        if self.min_clustersize < 1.0:
            minsamp = matrix.shape[0]*self.min_clustersize
            #print minsamp,matrix.shape, self.min_clustersize
        else:
            minsamp = self.min_clustersize

        def distances_select_first_non_id_neighbor(distances):
            x,y = distances.nonzero()
            _, idd = np.unique(x, return_index=True)

            """
            for i,e in enumerate(zip(list(x), list(y))):
                print e, distances[e]
                if i in idd:
                    print "!!!"
            print idd
            """
            return distances[ x[idd],y[idd]]


        #dists =  distances_select_NTH_non_id_neighbor(dist,2)
        dists =  distances_select_first_non_id_neighbor(dist)
        #dist = np.median(dists)
        dists=np.sort(dists)
        idx=int(len(dists)*self.dbscan_range)
        dist=dists[idx]
        if self.debug:
            print "name_subgraph: choosing dist %d of %d" % (idx, len(dists))



        # get the clusters
        scan = DBSCAN(eps=dist, min_samples=minsamp)
        return scan.fit_predict(matrix)

    def predict(self, matrix):
        return self.cluster_classifier.predict(matrix)

    def fit(self, subgraphs):
        # delete duplicates
        print "got %d subgraphs" % len(subgraphs)
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
                draw.graphlearn(graphclusters[cid][:5], size=3)


class ClusterClassifier_keepduplicates():                                                 #!!!!!!!!!!!!!

    def __init__(self,debug=False, vectorizer=Vectorizer(),min_clustersize=2,dbscan_range=.6):
        self.debug=debug
        self.vectorizer=vectorizer
        self.min_clustersize=min_clustersize
        self.dbscan_range=dbscan_range

    def cluster_subgraphs(self, matrix, nth_neighbor=1):
        '''
        get the median distance to the NTH neighbor with NN
        use that distance to cluster with scan
        '''
        n_neighbors = min(100, matrix.shape[0])
        if n_neighbors < 100 and self.debug:
            print 'name_subgraphs: there are %d graphs to cluster' % matrix.shape[0]

        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        neigh.fit(matrix)
        dist, indices = neigh.kneighbors(matrix)

        def distances_select_NTH_non_id_neighbor(distances,N):
            x,y = distances.nonzero()
            print x
            state=(-1,-1)
            idd=[]
            for idx,xx in enumerate(x):
                if xx!=state[1]:   # we see a new letter -> state is 1
                    state = (1,xx)
                elif state[0] == 1:
                    idd.append(idx)
                    state=(2,xx)

            return distances[ x[idd],y[idd]]

        def distances_select_first_non_id_neighbor(distances):
            x,y = distances.nonzero()
            _, idd = np.unique(x, return_index=True)

            """
            for i,e in enumerate(zip(list(x), list(y))):
                print e, distances[e]
                if i in idd:
                    print "!!!"
            print idd
            """
            return distances[ x[idd],y[idd]]


        #dists =  distances_select_NTH_non_id_neighbor(dist,2)
        dists =  distances_select_first_non_id_neighbor(dist)
        #dist = np.median(dists)
        dists=np.sort(dists)
        idx=int(len(dists)*self.dbscan_range)
        dist=dists[idx]
        if self.debug:
            print "name_subgraph: choosing dist %d of %d" % (idx, len(dists))


        if self.min_clustersize < 1.0:
            minsamp = max(int(matrix.shape[0]*self.min_clustersize),2)
            logger.debug( "minimum cluster size is: %d" % minsamp )
        else:
            minsamp = self.min_clustersize


        # get the clusters
        scan = DBSCAN(eps=dist, min_samples=minsamp)
        return scan.fit_predict(matrix)

    def predict(self, matrix):
        return self.cluster_classifier.predict(matrix)

    def fit(self, subgraphs):
        # delete duplicates
        #subgraphs = utils.unique_graphs_graphlearn_graphhash(subgraphs)
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
                print "cluster: %d  len: %d" % (cid, len(graphclusters[cid]))
                #subgraphs = utils.unique_graphs_graphlearn_graphhash(subgraphs)
                draw.graphlearn(utils.unique_graphs_graphlearn_graphhash(graphclusters[cid])[:5],edge_label='label', size=3)



