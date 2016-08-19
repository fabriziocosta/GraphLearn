"""
here we learn how to generate a graph minor such that
later transform:graph->(graph,graphminor)


fit will first train a oneclass estimator on the input graphs.

then we find out which nodes will be grouped together to an abstract node.
this can be done via group_score_threshold, where high scoring nodes will be placed together.
the alternative is to look at all the estimator scores for each node and k-mean them.
with both strategies, only groups of a certain minimum size will be contracted to a minor node.

in the end all groups can be automatically clustered and named by their cluster(name_cluster).

THIS IS A TEST I AM TRYING TO MAKE THIS MORE EZ AND EZ TO TEST


"""

import logging
from graphlearn.minor.autoabstract.name_subgraphs import ClusterClassifier, unique_graphs
import abstractor
import graphlearn.utils.draw as draw
from graphlearn.estimate import ExperimentalOneClassEstimator
from graphlearn.transform import GraphTransformer
logger = logging.getLogger(__name__)
from eden.graph import Vectorizer
import annotate

class GraphMinorTransformer(GraphTransformer):
    def __init__(self,
                 vectorizer=Vectorizer(complexity=3),
                 estimator=ExperimentalOneClassEstimator(),
                 group_min_size=2,
                 group_max_size=5,
                 # cluster_min_members=0,
                 # cluster_max_members=-1,
                 group_score_threshold=0.4,
                 debug=False,
                 # subgraph_cluster=,
                 cluster_classifier=ClusterClassifier(),
                 # save_graphclusters=False,
                 multiprocess=True,
                 layer=0):

        self.vectorizer = vectorizer
        self.estimator = estimator
        self.max_size = group_max_size
        self.min_size = group_min_size
        self.score_threshold = group_score_threshold
        self.debug = debug
        # self.subgraph_cluster=subgraph_cluster
        self.cluster_classifier = cluster_classifier
        # self.save_graphclusters=save_graphclusters
        # self.cluster_min_members=cluster_min_members
        # self.cluster_max_members=cluster_max_members
        self.layer = layer
        self.multiprocess = multiprocess


    def prepfit(self):
        self.abstractor = abstractor.GraphToAbstractTransformer(
            score_threshold=self.score_threshold,
            min_size=self.min_size,
            max_size=self.max_size,
            debug=self.debug,
            estimator=None,
            layer=self.layer,
            vectorizer=self.vectorizer)

        self.annotator= annotate.Annotator()

    def fit(self, graphs, fit_transform=False):
        '''
        TODO: be sure to set the self.cluster_ids :)

        Parameters
        ----------
        graphs

        Returns
        -------

        '''
        #  PREPARE
        graphs = list(graphs)
        if graphs[0].graph.get('expanded', False):
            raise Exception('give me an unexpanded graph')
        self.prepfit()
        # info
        if self.debug:
            print 'minortransform fit. input after select layer'
            draw.graphlearn(graphs[:3], contract=False, size=4, vertex_label='label')


        # annotate graphs and GET SUBGRAPHS
        graphs = self.annotator.fit_transform(graphs)
        subgraphs = list(self.abstractor.get_subgraphs(graphs))

        # FILTER UNIQUES AND TRAIN THE CLUSTERER
        self.cluster_classifier.fit(subgraphs)
        self.abstractor.nameestimator = self.cluster_classifier

        # annotating is super slow. so in case of fit_transform i can save that step
        if fit_transform:
            return self.transform(graphs)

    def fit_transform(self, inputs):
        return self.fit(inputs, fit_transform=True)



    def transform(self, graphs):
        '''
        Parameters
        ----------
        inputs: [graph]

        Returns
        -------
            [(edge_expanded_graph, minor),...]
        '''

        graphs = self.annotator.transform(graphs)
        result = self.abstractor.transform(graphs)
        if self.debug:
            print 'minortransform  transform. the new layer  '
            draw.graphlearn(result[:3], contract=False, size=6, vertex_label='contracted')
        return result


