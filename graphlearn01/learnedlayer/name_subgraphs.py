import eden
import numpy as np
import sklearn

from sklearn.linear_model import SGDClassifier
import copy
import sklearn
from graphlearn01 import utils
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from eden.graph import Vectorizer
import hashlib
from graphlearn01.utils import draw
from collections import Counter
import numpy as np





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

    def predict(self, matrix, _):
        return self.cluster_classifier.predict(matrix)

    def fit(self, subgraphs):
        # delete duplicates
        #print "got %d subgraphs" % len(subgraphs)
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
        self.cluster_classifier_failed=False

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

    def predict(self, matrix,_):
        if self.cluster_classifier_failed:
            return  [1]*len(_)
        return self.cluster_classifier.predict(matrix)


    def fit(self, subgraphs):
        # delete duplicates
        #subgraphs = utils.unique_graphs_graphlearn_graphhash(subgraphs)
        try:
            matrix = self.vectorizer.transform(subgraphs)
            cluster_ids = self.cluster_subgraphs(matrix)

            self.cluster_classifier = SGDClassifier()
            self.cluster_classifier.fit(matrix, cluster_ids)

            if self.debug:
                graphclusters = defaultdict(list)
                for i, cluster_id in enumerate(cluster_ids):
                    # if cluster_id not in self.ignore_clusters:
                    graphclusters[cluster_id].append(subgraphs[i])
                # info
                logger.debug('num clusters: %d' % max(cluster_ids))
                logger.debug(eden.util.report_base_statistics(cluster_ids).replace('\t', '\n'))

                idcounter=Counter(cluster_ids)
                # ok now we want to print the INFO from above
                for cid, count in idcounter.most_common():

                    # this dist stuff is untested btw.. the idea was to order the graphs s.th the center one comes first
                    uniquegraphs = utils.unique_graphs_graphlearn_graphhash(graphclusters[cid])
                    dists = sklearn.metrics.pairwise.pairwise_distances(self.vectorizer.transform(copy.deepcopy(uniquegraphs)))
                    argmins = np.min(dists, axis=0)
                    posstuff = [ (e,i) for i,e in enumerate(argmins) ]
                    posstuff.sort()
                    res= [e[1] for e in posstuff[:5] ]
                    print "cluster: %d  len: %d" % (cid, len(graphclusters[cid]))
                    #subgraphs = utils.unique_graphs_graphlearn_grahhash(subgraphs)
                    draw.graphlearn([uniquegraphs[i] for i in res],edge_label='label', size=3)
        except:
            self.cluster_classifier_failed=True

class ClusterClassifier_fake():

    def __init__(self,debug=False, vectorizer=Vectorizer(),min_clustersize=2,dbscan_range=.6):
        self.debug=debug
        self.vectorizer=vectorizer
        self.min_clustersize=min_clustersize
        self.dbscan_range=dbscan_range

    def fit(self, subgraphs):
        subgraphs_by_keys = defaultdict(list)
        for e in subgraphs:
            subgraphs_by_keys[e.graph['interface_hash']].append(e)

        self.clusters={}
        for k,v in subgraphs_by_keys.iteritems():
            if len(v) >= self.min_clustersize:
                self.clusters[k]=1


        if self.debug:
            histodata=[]
            for k,v in subgraphs_by_keys.iteritems():
                if len(v) >= self.min_clustersize:
                    noduplen = len( utils.unique_graphs_graphlearn_graphhash(copy.deepcopy(v)))
                    histodata.append((len(v), noduplen))
            histodata.sort()

            print 'there are %d subgraphs with interfaces, those unter min_clustersize are omitted' % len(subgraphs)

            a,b = zip(*histodata)
            draw.plot_charts2(a,b, datalabels=["all subgraphs","removed duplicates"] ,xlabel='Interfaces', ylabel="Count",  log_scale=False )




    def predict(self, matrix, subgraphs):
        res=[]
        for subgraph in subgraphs:
            if subgraph.graph['interface_hash'] in self.clusters:
                res.append( subgraph.graph['interface_hash'])
            res.append(-1)
        return res


class ClusterClassifier_keepduplicates_interfaced():                                                 #!!!!!!!!!!!!!

    def __init__(self,debug=False, vectorizer=Vectorizer(),min_clustersize=2,dbscan_range=.6):
        self.debug=debug
        self.vectorizer=vectorizer
        self.min_clustersize=min_clustersize
        self.dbscan_range=dbscan_range




    def fit(self, subgraphs):

        subgraphs_by_keys = defaultdict(list)
        for e in subgraphs:
            subgraphs_by_keys[e.graph['interface_hash']].append(e)


        self.classifiers={}
        for e in subgraphs_by_keys.keys():
            if len(subgraphs_by_keys[e]) > 10:
                classifier = ClusterClassifier_keepduplicates(self.debug, self.vectorizer, self.min_clustersize,self.dbscan_range)
                classifier.fit(subgraphs_by_keys[e])
                self.classifiers[e]=classifier

        if self.debug:
            histodata=[]
            for k,v in subgraphs_by_keys.iteritems():
                if len(v) >= self.min_clustersize:
                    noduplen = len( utils.unique_graphs_graphlearn_graphhash(copy.deepcopy(v)))
                    histodata.append((len(v), noduplen))
            histodata.sort()

            print 'there are %d subgraphs with interfaces, those unter min_clustersize are omitted' % len(subgraphs)

            a,b = zip(*histodata)
            draw.plot_charts2(a,b, datalabels=["all subgraphs","removed duplicates"] ,xlabel='Interfaces', ylabel="Count",  log_scale=False )


    def predict(self, matrix, subgraphs):
        res=[]

        for vec,subgraph in zip(matrix,subgraphs):


            # we append -1
            appendvalue=-1
            #  except when we find the interface in the classifiers
            if subgraph.graph['interface_hash'] in self.classifiers:
                cluster = self.classifiers[subgraph.graph['interface_hash']].predict(vec,[subgraph])[0]
                if cluster != -1:
                    try:
                        appendvalue =  "%d#%d" % ( cluster, subgraph.graph['interface_hash'])
                    except:
                        print 'name subgraph keep interf has a problem', subgraph.graph['interface_hash']

            res.append(appendvalue)

        return res



class ClusterClassifier_soft_interface():
    '''keeps duplicates'''

    def __init__(self,debug=False, vectorizer=Vectorizer(),min_clustersize=2,dbscan_range=.6, interfaceweight=-2):
        self.debug=debug
        self.vectorizer=vectorizer
        self.min_clustersize=min_clustersize
        self.dbscan_range=dbscan_range
        self.interfaceweight=interfaceweight # for vectorization

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


        def distances_select_first_non_id_neighbor(distances):
            x,y = distances.nonzero()
            _, idd = np.unique(x, return_index=True)
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

    def predict(self, _, graphs):

        matrix = self.trixify(graphs)
        return self.cluster_classifier.predict(matrix)

    def trixify(self,graphs):


        '''there are cips comming in, cips have something something'''
        for g in graphs:
            for n,d in g.nodes(data=True):
                if d.get('interface',False):
                    d['weight'] = 2**self.interfaceweight
                else:
                    d['weight'] = 1.0

        #except:
        #    #draw.graphlearn(subgraphs, contract= False)
        '''
        def ginfo(g):
            print utils.ascii.nx_to_ascii(g)
            print g.nodes(data=True)
            print g.edges(data=True)

        for g in graphs:
            for n,d in g.nodes(data=True):
                if 'label' not in d:
                    ginfo(g)
            ginfo(g)
            self.vectorizer.transform([g])
        '''

        data=self.vectorizer.transform(graphs)
        #        #draw.debug(e)
        return data


    def fit(self, subgraphs):
        #print "asdasd"
        #import structout as so
        matrix =  self.trixify(subgraphs) #self.vectorizer.transform(subgraphs)
        #so.gprint(subgraphs[:3],label='weight')
        cluster_ids = self.cluster_subgraphs(matrix)

        self.cluster_classifier = SGDClassifier()
        self.cluster_classifier.fit(matrix, cluster_ids)

        if self.debug:
            graphclusters = defaultdict(list)
            for i, cluster_id in enumerate(cluster_ids):
                # if cluster_id not in self.ignore_clusters:
                graphclusters[cluster_id].append(subgraphs[i])
            # info
            logger.debug('num clusters: %d' % max(cluster_ids))
            logger.debug(eden.util.report_base_statistics(cluster_ids).replace('\t', '\n'))

            idcounter=Counter(cluster_ids)
            # ok now we want to print the INFO from above
            for cid, count in idcounter.most_common():

                # this dist stuff is untested btw.. the idea was to order the graphs s.th the center one comes first
                uniquegraphs = utils.unique_graphs_graphlearn_graphhash(graphclusters[cid])
                dists = sklearn.metrics.pairwise.pairwise_distances(self.vectorizer.transform(copy.deepcopy(uniquegraphs)))
                argmins = np.min(dists, axis=0)
                posstuff = [ (e,i) for i,e in enumerate(argmins) ]
                posstuff.sort()
                res= [e[1] for e in posstuff[:5] ]
                print "cluster: %d  len: %d" % (cid, len(graphclusters[cid]))
                #subgraphs = utils.unique_graphs_graphlearn_grahhash(subgraphs)
                draw.graphlearn([uniquegraphs[i] for i in res],edge_label='label', size=3)



