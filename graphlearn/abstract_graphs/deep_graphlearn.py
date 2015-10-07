





from graphlearn.graphtools import GraphWrapper
from graphlearn.graphlearn import GraphLearnSampler



class deepsample(GraphLearnSampler):


    def fit(self, input, n_jobs=-1, nu=.5, batch_size=10):
        """
          use input to fit the grammar and fit the estimator
        """

        graphmanagers = self.preprocessor.fit_transform(input)

        self.estimatorobject.fit(graphmanagers,
                                                  vectorizer=self.vectorizer,
                                                  nu=nu,
                                                  n_jobs=n_jobs,
                                                  random_state=self.random_state)
        tempest= EstimatorWrapper()
        tempest.fit(graphmanagers,
                                                  vectorizer=self.vectorizer,
                                                  nu=nu,
                                                  n_jobs=n_jobs,
                                                  random_state=self.random_state)


        self.lsgg.fit(graphmanagers, n_jobs, batch_size=batch_size)

        '''
        HOW TO TRAIN NEW CORES?
        make a sampler
        with: estimator as estimator, interface-groups as input, dat filter for cip choosing
        '''


        def entitycheck(g,nodes):
            if type(nodes) is not list:
                nodes=[nodes]
            for e in nodes:
                if 'interface' in g.node[e]:
                    return False
            return True


        prod=self.lsgg.productions
        for k in prod.keys():
            graphs=prod[k].values()
            sampler=GraphLearnSampler(estimator=tempest,node_entity_check=entitycheck)
            graphs=[ GraphWrapper(graph, self.vectorizer) for graph in graphs ]
            sampler.lsgg.fit(graphs)
            sampler.sample



