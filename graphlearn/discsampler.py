

from graphlearn import GraphLearnSampler, LocalSubstitutableGraphGrammar
import itertools
import networkx as nx
from scipy.sparse import csr_matrix ,vstack
from sklearn.neighbors import NearestNeighbors
import numpy as np
import heapq
from eden.util import fit
from eden.graph import Vectorizer



class discsampler():
    '''
    ok new plan is this:
    '''

    def  __init__(self):
        self.vectorizer= Vectorizer()



    def fit(self, iter_pos, iter_neg):

        # getting the sampler ready:
        self.sampler = GraphLearnSampler()
        iter_pos, pos, pos_ = itertools.tee(iter_pos,3)
        self.estimator = self.sampler.estimator.fit2(iter_pos, iter_neg, self.vectorizer)
        self.sampler.estimator= self.estimator
        self.sampler.fitgrammar(pos )

        # calculating avg distance
        X = self.vectorizer.transform(pos_)
        



    def calculatedistancemeassure(self):
        pass
        #while not done
            # PICK the 10 closest to the hyperplane and sample (score,graph)

            # check those 10 and add all the valid points to some working set








class OLD:
    def sample(self, graph_iter,
               batch_size=2,
               n_jobs=-1,
               n_steps=100,
               select_cip_max_tries = 100,
               annealing_factor = 1.0,
               queue_chunk_size = 10,
               radius = 0.15,
               create_n_samples = 100,
               sample_tries = 30, # 30 is default according to algo
               ):




        self.sample_tries= sample_tries
        # initialize
        new_graphs= []

        start_graphs, graphlist_ = itertools.tee(graph_iter)
        vectors = self.vectorizer.transform(graphlist_)
        queue = start_graphs

        nbrs= NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vectors)
        #dist, indices= nbrs.kneighbors(X)


        new_vectors = None

        # l00p1ng
        while len(new_graphs) < create_n_samples and queue:


            # take from the queue and start sampling
            graphs= itertools.islice(queue,0,queue_chunk_size)

            print 'start '+str(len(new_graphs))
            for result_list in super(discsampler,self).sample( graphs ,sampling_interval=9999,
                                batch_size=batch_size,n_jobs=n_jobs, n_steps=n_steps,
                                same_core_size=False,
                                same_radius=False,
                                annealing_factor = annealing_factor ,
                                select_cip_max_tries=select_cip_max_tries,similarity=(1-radius)
                                ):

                # lets see what we created:
                # we need to test everything
                # and see what we want to keep
                for graph in result_list:
                    if graph == None: # batching creates nones
                        continue
                    vectorized=self.vectorizer._convert_dict_to_sparse_matrix(
                        self.vectorizer._transform(0, nx.Graph(graph)))
                    
                    # check with the old ones:
                    d,i = nbrs.kneighbors(vectorized)
                    dist= i.sum()
                    if i < radius:
                        print 'rejected, neighbor too close'
                        continue
                    # check with the new ones:
                    nogood=False

                    if new_vectors != None:
                        list=vectorized.dot( new_vectors.T ).todense().tolist()
                        for e in list[0]:
                            if (1-e) < radius:
                                print 'rejected, newvec too close'
                                nogood=True
                                break
                    if nogood:
                        continue
                    # if we keep them, we put them in the queue
                    queue = itertools.chain(queue,graph)
                    new_graphs.append(graph)
                    if new_vectors==None:
                        new_vectors= vectorized
                    else:
                        new_vectors = vstack([vectorized,new_vectors], format='csr')

        return new_graphs




    def _stop_condition(self, graph):
        '''
        '''
        if self.similarity > 0:
            if self.step == 0:
                self.vectorizer._reference_vec = \
                    self.vectorizer._convert_dict_to_sparse_matrix(
                        self.vectorizer._transform(0, nx.Graph(graph)))
            else:
                x = self.vectorizer._convert_dict_to_sparse_matrix(
                        self.vectorizer._transform(0,nx.Graph(graph)))
                similarity  = self.vectorizer._reference_vec.dot(x.T).todense().sum()

                if  similarity < self.similarity:
                    raise Exception('similarity stop condition reached')


    # this will yield up to 30 graphs... graph
    def _sample(self,input):
        res_list= []
        for x in xrange(self.sample_tries):
            inp=nx.Graph(input)
            res_list.append( super(discsampler,self)._sample(inp)[0] )
        return res_list

