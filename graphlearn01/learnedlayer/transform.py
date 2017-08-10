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

logger = logging.getLogger(__name__)
from name_subgraphs import ClusterClassifier
import abstractor
import graphlearn01.utils.draw as draw
import graphlearn01.utils.ascii as ascii
from graphlearn01.estimate import ExperimentalOneClassEstimator
from graphlearn01.transform import GraphTransformer
# import graphlearn.utils as utils
import eden
import annotate
import graphlearn01
import graphlearn01.utils.ascii as ascii

class GraphMinorTransformer(GraphTransformer):

    def toggledebug(self):
        self.debug= not self.debug
        self.annotator.debug= not self.debug
        self.cluster_classifier.debug = not self.debug

    def __init__(self,
                 vectorizer=eden.graph.Vectorizer(complexity=3),
                 estimator=ExperimentalOneClassEstimator(),
                 group_min_size=3,
                 group_max_size=6,
                 # cluster_min_members=0,
                 # cluster_max_members=-1,
                 group_score_threshold=0.4,
                 debug=False,
                 debug_rna=False,
                 # subgraph_cluster=,
                 cluster_classifier=ClusterClassifier(debug=False,vectorizer=eden.graph.Vectorizer()),
                 # save_graphclusters=False,
                 multiprocess=True,
                 num_classes=2,
                 layer=0):

        self.vectorizer = vectorizer
        self.estimator = estimator
        self.max_size = group_max_size
        self.min_size = group_min_size
        self.score_threshold = group_score_threshold
        self.debug = debug
        self.debug_rna=debug_rna
        # self.subgraph_cluster=subgraph_cluster
        self.cluster_classifier = cluster_classifier
        # self.save_graphclusters=save_graphclusters
        # self.cluster_min_members=cluster_min_members
        # self.cluster_max_members=cluster_max_members
        self.layer = layer
        self.multiprocess = multiprocess
        self.num_classes = num_classes

    def prepfit(self):
        self.cluster_classifier.debug = self.debug
        self.abstractor = abstractor.GraphToAbstractTransformer(
            score_threshold=self.score_threshold,
            min_size=self.min_size,
            max_size=self.max_size,
            debug=self.debug,
            estimator=None,
            layer=self.layer,
            vectorizer=self.vectorizer)

        self.annotator = annotate.Annotator(vectorizer=self.vectorizer,multiprocess=False)

    def fit(self, graphs, graphs_neg=[], fit_transform=False):
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
        graphs_neg = list(graphs_neg)
        if graphs[0].graph.get('expanded', False):
            raise Exception('give me an unexpanded graph')
        self.prepfit()


        # info
        if self.debug:
            print 'minortransform_fit'
            if self.debug_rna:
                draw.graphlearn(graphs[:5], contract=False, size=12, vertex_label='label')
            else:
                draw.graphlearn(graphs[:5], contract=False, size=5, vertex_label='label')

        # annotate graphs and GET SUBGRAPHS
        graphs = self.annotator.fit_transform(graphs, graphs_neg)

        # draw.graphlearn([graphs[0], graphs_neg[-1]], vertex_label='importance')
        # info
        if self.debug:
            print 'minortransform_scores'
            if self.debug_rna:
                draw.graphlearn(graphs[:5], contract=False, size=12, vertex_label='importance')
            else:
                for g in graphs[:5]:
                    for e,d in g.nodes(data=True):
                        d['importance_sd'] = d['importance'][0]
                draw.graphlearn(graphs[:5], contract=False, size=5, vertex_label='importance_sd')
                #ascii.printrow(graphs[:3],size=14)
                #ascii.printrow(graphs[3:6],size=14)
            # vertex_color='importance', colormap='inferno')



        subgraphs = list(self.abstractor.get_subgraphs(graphs))
        if len(subgraphs) ==0:
            print "FFFUUUUCCCCKKKK learnedlayer transform py"

        # FILTER UNIQUES AND TRAIN THE CLUSTERER
        self.cluster_classifier.fit(subgraphs)
        self.abstractor.nameestimator = self.cluster_classifier

        # annotating is super slow. so in case of fit_transform i can save that step
        if fit_transform:
            return self.transform(graphs)

    def fit_transform(self, inputs, inputs_neg=[]):
        return self.fit(inputs, inputs_neg, fit_transform=True)

    def re_transform_single(self, graph):

        # graphlearn is giving me expanded graphs afaik
        return self.transform([eden.graph._revert_edge_to_vertex_transform(graph)])[0]

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

        #print "I AM HERE LCIK ME"
        #for gg in graphs: 
        #    print gg.nodes(data=True)
        #    draw.debug(gg)

        #print 'graphs %d' % len(graphs)
        result = self.abstractor.transform(graphs)

        #print 'res %d' % len(result)


        if self.debug:
            pass
            #print 'minortransform  transform. the new layer  '
            #if self.debug_rna:
            #    draw.graphlearn(result[:5], contract=False, size=12, vertex_label='contracted' )
            #else:
            #    draw.graphlearn(result[:5], contract=False, size=6, vertex_label='contracted' )
        return result
