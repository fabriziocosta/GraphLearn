






from graphlearn.graphlearn import GraphLearnSampler



class deepsample(GraphLearnSampler):



    def fit(self, input, n_jobs=-1, nu=.5, batch_size=10):
        """
          use input to fit the grammar and fit the estimator
        """

        graphmanagers = self.fit_to_graphmanager(input)


        self.estimatorobject.fit(graphmanagers,
                                                  vectorizer=self.vectorizer,
                                                  nu=nu,
                                                  n_jobs=n_jobs,
                                                  random_state=self.random_state)
        self.lsgg.fit(graphmanagers, n_jobs, batch_size=batch_size)






'''
a wrapper that uses base_graph()
'''
from graphlearn.estimatorwrapper import EstimatorWrapper
class esti(EstimatorWrapper):
    def unwrap(self,graphmanager):
        graph = graphmanager.base_graph().copy()
        if type(graph) == nx.DiGraph:
            graph=nx.Graph(graph)
        return graph