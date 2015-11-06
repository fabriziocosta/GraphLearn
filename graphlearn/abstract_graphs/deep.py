





from graphlearn.graph import Wrapper as GraphWrap
from graphlearn.graphlearn import GraphLearnSampler
from graphlearn.estimator import Wrapper as EstiWrap
from graphlearn.coreinterfacepair import CoreInterfacePair
from graphlearn.utils import draw

class DeepSampler(GraphLearnSampler):


    def fit(self, input, grammar_n_jobs=-1, grammar_batch_size=10, train_min_size=None):
        """
          use input to fit the grammar and fit the estimator
        """
        self.preprocessor.set_param(self.vectorizer)

        graphmanagers = self.preprocessor.fit_transform(input,self.vectorizer)

        self.estimatorobject.fit(graphmanagers,
                                 vectorizer=self.vectorizer,
                                 nu=nu,
                                 grammar_n_jobs=grammar_n_jobs,
                                 random_state=self.random_state)

        self.lsgg.fit(graphmanagers, grammar_n_jobs, grammar_batch_size=grammar_batch_size)


        tempest= EstiWrap(nu=.5,  grammar_n_jobs=grammar_n_jobs)
        tempest.fit(graphmanagers,
                    vectorizer=self.vectorizer,

                    random_state=self.random_state)




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

        for i, interface_hash in enumerate(prod.keys()):

            if prod[interface_hash] < train_min_size:
                continue
            print "################################# new ihash"
            # for all the interface buckets
            cips=prod[interface_hash].values()
            sampler=GraphLearnSampler(estimator=tempest,node_entity_check=entitycheck)
            graphs_wrapped=[ GraphWrap(cip.graph, self.vectorizer) for cip in cips ]
            graphs=[ cip.graph for cip in cips ]

            sampler.lsgg.fit(graphs_wrapped)
            sampler.preprocessor.fit(0,self.vectorizer)
            sampler.postprocessor.fit(sampler.preprocessor)
            r=sampler.sample(graphs, max_size_diff=0, select_cip_max_tries=100, quick_skip_orig_cip=False,
                             improving_linear_start=.2, improving_threshold=.6)

            # get graphs and sample them
            r= list(r)

            for j, raw_graph in enumerate(r):
                # for each resulting graph
                raw_graph.graph.pop('graph',None)
                score= tempest.score(raw_graph)
                if score > tempest.score(cips[j].graph):
                    # check if the score is good enough, then add to grammar
                    self.lsgg.productions[interface_hash][score]=CoreInterfacePair(
                         interface_hash=cips[j].interface_hash,
                         core_hash=score,
                         graph=raw_graph,
                         radius=cips[j].radius,
                         thickness=cips[j].thickness,
                         core_nodes_count=len(raw_graph),
                         count=1,
                         distance_dict=cips[j].distance_dict)
                    print 'new graph:',score
                    draw.graphlearn(raw_graph)




