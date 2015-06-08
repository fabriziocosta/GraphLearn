from graphlearn import GraphLearnSampler, LocalSubstitutableGraphGrammar
import itertools
import networkx as nx
from scipy.sparse import csr_matrix, vstack
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LSHForest
import numpy as np
import heapq
from eden.util import fit
from eden.graph import Vectorizer
import copy
import heapq

from utils.draw import draw_grammar
class DiscSampler():
    '''
    '''

    def __init__(self):
        # this is mainly for the forest. the sampler uses a different vectorizer
        self.vectorizer = Vectorizer(nbits=14)

    def get_heap_and_forest(self, iter, k):
        '''
        so we create the heap and the forest...
        heap is (dist to hyperplane, count, graph)
        and the forest ist just a nearest neighbor from sklearn
        '''
        iter, iter2 = itertools.tee(iter)
        X = self.vectorizer.transform(iter)
        forest = LSHForest()
        forest.fit(X)
        heap = []
        print 'got forest'
        for vector, graph in itertools.izip(X, iter2):
            graph2 = nx.Graph(graph)
            heapq.heappush(heap, (
            self.sampler.estimator.predict_proba(self.sampler.vectorizer.transform_single(graph2))[0][1], k + 1, graph))
        print 'got heap'
        distances, unused = forest.kneighbors(X, n_neighbors=2)
        distances = [a[1] for a in distances]  # the second element should be the dist we want
        avg_dist = distances[len(distances) / 2]  # sum(distances)/len(distances)
        print 'got dist'
        return heap, forest, avg_dist

    def sample_graphs(self, graphiter, iter_neg, radius, how_many, check_k, heap_chunk_size=10):

        # some initialisation,
        # creating samper
        # setup heap and forest
        graphiter, iter2 = itertools.tee(graphiter)
        self.fit_sampler(iter2, iter_neg)
        heap, forest, avg_dist = self.get_heap_and_forest(graphiter, check_k)
        # heap should be like   (hpdist, count, graph)
        radius = radius * avg_dist

        # so lets start the loop1ng
        result = []
        while heap and len(result) < how_many:

            # pop all the graphs we want
            todo = []
            for i in range(heap_chunk_size):
                if heap:
                    todo.append(heapq.heappop(heap))

            # let the sampler do the sampling
            graphz = [e[2] for e in todo]
            work = self.sampler.sample(graphz,
                                       batch_size=3,
                                       n_jobs=1,
                                       n_steps=30,
                                       select_cip_max_tries=100,
                                       accept_annealing_factor=.5,
                                       generatormode=False,
                                       same_core_size=False 
                                       )

            # lets see, we need to take care of
            # = the initialy poped stuff
            # - increase and check the counter, reinsert into heap
            # = the new graphs
            # put them in the heap and the forest
            for graph, task in zip(work, todo):
                graphlist = graph.graph['sampling_info']['graphs_history']
                print 'rez:', graphlist, task
                for graph in graphlist:
                    # check distance from created instances
                    x = self.vectorizer.transform_single(graph)
                    dist, void = forest.kneighbors(x, 1)
                    dist = sum(dist)
                    # is the distance ok?
                    # if so, insert into forest and heap
                    if radius < dist < radius * 2:
                        forest.partial_fit(x)
                        heapq.heappush(heap, (graph.graph['score'], 0, graph))
                        print 'heap'
                    print 'cant heap', radius, dist
                # taking care of task graph
                # put in result list if necessary
                if task[1] < check_k < task[1] + len(graphlist):
                    result.append(task[2])
                    print 'found sth'
                # go back to the heap!
                heapq.heappush(heap, (task[0], task[1] + len(graphlist), task[2]))

        return result

    def fit_sampler(self, iter_pos, iter_neg):

        # getting the sampler ready:
        vectorizer= Vectorizer(complexity=3, nbits=20)
        self.sampler = MySampler(vectorizer=vectorizer)
        iter_pos, pos, pos_ = itertools.tee(iter_pos, 3)
        self.estimator = self.sampler.estimatorobject.fit_2(iter_pos, iter_neg, vectorizer)
        print 'got estimeetaaa'
        self.sampler.estimator = self.estimator
        self.sampler.fit_grammar(pos)
        print 'got grammar:grammar is there oO'
        #draw_grammar(self.sampler.local_substitutable_graph_grammar.grammar,n_productions=5)
        


class MySampler(GraphLearnSampler):
    def _stop_condition(self, graph):
        '''
        i accept 2 versions oOo
        '''
        is_new = True
        for gr in self.sample_path:
            if gr._score == graph._score:
                is_new = False
        if is_new:
            self.sample_path.append(graph)

        if len(self.sample_path) > 3:
            raise ('stop condition reached')

    def _sample_path_append(self, graph):
        pass

        # this will yield up to 30 graphs... graph
        # def _sample(self,input):
        #    res_list= []
        #    for x in xrange(3): # hijacking similarity oO
        #        inp=nx.Graph(input)
        #        res_list+= GraphLearnSampler._sample(self,inp).graph['sampling_info']['graphs_history'][1:-1]
        #    return res_list
