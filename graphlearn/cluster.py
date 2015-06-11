from graphlearn import GraphLearnSampler, LocalSubstitutableGraphGrammar
import itertools
import networkx as nx
from sklearn.neighbors import LSHForest
from eden.util import selection_iterator

from sklearn.metrics.pairwise import cosine_distances as distance


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
                                                                                nbit=self.nbit,
                                                                                node_entity_check=self.node_entity_check)
        self.local_substitutable_graph_grammar.fit(G_pos, n_jobs)

    def get_nearest_neighbor_iterable(self, graphlist, start_graphs, start_is_subset=True):

        # vectorize all
        graphlist, graphlist_ = itertools.tee(graphlist)
        X = self.vectorizer.transform(graphlist_)

        start_graphs, graphlist_ = itertools.tee(start_graphs)
        Y = self.vectorizer.transform(graphlist_)
        forest = LSHForest()
        forest.fit(X)
        distances, indices = forest.kneighbors(Y, n_neighbors=2)

        # we just assume that this is short...
        start_graphs = list(start_graphs)
        index = 0
        if start_is_subset:
            index += 1

        matches = [(indices[i, index], i, distances[i, index]) for i in range(len(indices))]
        matches.sort()

        for index, graph in enumerate(selection_iterator(graphlist, [a[0] for a in matches])):
            yield ((graph, start_graphs[matches[index][1]], X[matches[index][0]]))

    '''
        # iterate over graphs
        graphlist, graphlist_ = itertools.tee(graphlist)
        for i, g in enumerate(graphlist):

            # compare to all other graphs and see who the NN is :)
            gl, graphlist_ = itertools.tee(graphlist_)
            best_sim = 0.0
            NN = 0
            for i2, g2 in enumerate(gl):
                sim = X[i].dot(X[i2].T).todense()[0][0]
                # print sim # just to make sure..
                if sim > best_sim and i != i2:
                    best_sim = sim
                    NN = g2
            yield (g, NN, X[i2])
    '''

    def _stop_condition(self, graph):

        if len(self.sample_path) == 1:
            self.sample_path.append(self.goal_graph)
        if 'score' in graph.__dict__:
            if graph.score > 0.99999:
                self._sample_notes += ';edge %d %d;' % (self.starthash, self.finhash)
                raise Exception('goal reached')

    def sample(self, graph_iter, targets, targets_in_graphs, **kwargs):

        graphiter = self.get_nearest_neighbor_iterable(graph_iter, targets, targets_in_graphs)
        # graphiter = itertools.islice(graphiter, doXgraphs)
        for e in super(cluster, self).sample(graphiter, **kwargs):
            yield e

    def _sample(self, g_pair):
        self.starthash = hash(g_pair[0])
        self.finhash = hash(g_pair[2])
        self.goal = g_pair[2]
        self.goal_graph = g_pair[1]
        self.goal_size = len(self.vectorizer._edge_to_vertex_transform(self.goal_graph))
        return super(cluster, self)._sample(g_pair[0])

    def _score(self, graph):
        if '_score' not in graph.__dict__:
            transformed_graph = self.vectorizer.transform_single(nx.Graph(graph))
            # slow so dont do it..
            # graph.score_nonlog = self.estimator.base_estimator.decision_function(transformed_graph)[0]

            # graph._score = self.goal.dot(transformed_graph.T).todense()[0][0].sum()
            graph._score = (1 - distance(transformed_graph, self.goal))[0, 0]

            # print graph._score
            # graph.score -= .007*abs( self.goal_size - len(graph) )
        return graph._score
