

from graphlearn import GraphLearnSampler, LocalSubstitutableGraphGrammar
import itertools

class cluster(GraphLearnSampler):

    '''
    ok here is the plan:

    1. train grammar as usual, no estimator needed.

    2. find NN

    3. the actual action might highjack the sample function.
        but instead of graph_iterator
        we give a graph_pair_iterator, start and goal graph,

    4. then we overwrite _sample to seperate start and goal again
        and use the original _sample with a changed scoring scheme

    5. draw result to see what actually happens

    6. ???

    7. we are done with the clustering process

    '''

    def fit(self, G_pos,
            core_interface_pair_remove_threshold=3,
            interface_remove_threshold=2,
            n_jobs=-1):
        """
          use input to fit the grammar and fit the estimator
        """
        # get grammar
        self.local_substitutable_graph_grammar = LocalSubstitutableGraphGrammar(self.radius_list, self.thickness_list,
                                                                                core_interface_pair_remove_threshold,
                                                                                interface_remove_threshold,
                                                                                nbit=self.nbit, node_entity_check=self.node_entity_check)
        self.local_substitutable_graph_grammar.fit(G_pos,n_jobs)



    def get_nearest_neighbor_iterable(self,graphlist):

        # vectorize all
        graphlist, graphlist_ = itertools.tee(graphlist)
        X = self.vectorizer.transform(graphlist_)

        graphlist,graphlist_ = itertools.tee(graphlist)
        for i,graph in enumerate(graphlist):
            gl,graphlist_ = itertools.tee(graphlist_)

            # compare to all other graphs and see who the nearest_neighbor_graph is :)
            best_sim = 0.0
            nearest_neighbor_graph = None

            for i2,graph2 in enumerate (gl):
                sim = X[i].dot(X[i2].T).todense()[0][0]
                #print sim # just to make sure..
                if sim > best_sim and i!=i2:
                    best_sim=sim
                    nearest_neighbor_graph = graph2

            yield (graph,nearest_neighbor_graph,X[i2])


    def _stop_condition(self, graph):

        if len(self.sample_path)==1:
            self.sample_path.append(self.goal_graph)
        if 'score' in graph.__dict__:
            if graph._score > 0.99999:
                self._sample_notes+=';edge %d %d;' % (self.starthash,self.finhash)
                raise Exception('goal reached')


    def sample(self, graph_iter, sampling_interval=9999,
               batch_size=10,
               n_jobs=0,
               n_steps=50,
               select_cip_max_tries = 20,
               annealing_factor= 1.0,
               doXgraphs=9999):


        graphiter = self.get_nearest_neighbor_iterable(graph_iter)
        graphiter = itertools.islice(graphiter,doXgraphs)
        for result_tuple in super(cluster,self).sample( graphiter ,sampling_interval=sampling_interval,
                            batch_size=batch_size,n_jobs=n_jobs, n_steps=n_steps,same_core_size=False,
                            annealing_factor = annealing_factor ,
                            select_cip_max_tries=select_cip_max_tries):
            yield result_tuple


    def _sample(self,g_pair):
        self.starthash = hash( g_pair[0] )
        self.finhash = hash(g_pair[2])
        self.goal = g_pair[2]
        self.goal_graph= g_pair[1]
        self.goal_size = len(self.vectorizer._edge_to_vertex_transform(self.goal_graph))
        return super(cluster,self)._sample(g_pair[0])


    def _score(self,graph):
        if not 'score' in graph.__dict__:
            transformed_graph = self.vectorizer.transform2(graph)
            # slow so dont do it..
            #graph.score_nonlog = self.estimator.base_estimator.decision_function(transformed_graph)[0]
            graph._score = self.goal.dot(transformed_graph.T).todense()[0][0].sum()
            # print graph.score
            graph._score -= .007*abs( self.goal_size - len(graph) )
        return graph._score

