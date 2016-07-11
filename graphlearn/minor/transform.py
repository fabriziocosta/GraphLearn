"""
here we learn how to generate a graph minor such that
later transform:graph->(graph,graphminor)


fit will first train a oneclass estimator on the input graphs.

then we find out which nodes will be grouped together to an abstract node.
this can be done via group_score_threshold, where high scoring nodes will be placed together.
the alternative is to look at all the estimator scores for each node and k-mean them.
with both strategies, only groups of a certain minimum size will be contracted to a minor node.

in the end all groups can be automatically clustered and named by their cluster(name_cluster).
"""
from eden.modifier.graph.structure import contraction
from collections import defaultdict
from graphlearn.estimate import OneClassEstimator
from graphlearn.transform import GraphTransformer
import graphlearn.utils.draw as draw
import networkx as nx
import logging
from itertools import izip
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from eden.util import report_base_statistics
logger = logging.getLogger(__name__)
from eden.graph import Vectorizer
from eden import graph as edengraphtools


class GraphMinorTransformer(GraphTransformer):
    def __init__(self,
                 vectorizer=Vectorizer(),
                 estimator=OneClassEstimator(),
                 group_min_size=2,
                 group_max_size=5,
                 cluster_min_members=0,
                 cluster_max_members=-1,
                 group_score_threshold=1.2,
                 debug=False,
                 subgraph_name_estimator=MiniBatchKMeans(n_clusters=5),
                 save_graphclusters=False):
        '''

        initial subgraphs are identified by threshold per default,
        you can also cluster those by setting group_score_classifier
        to eg sklearn.cluster.KMeans

        Parameters
        ----------
        vectorizer: eden vectorizer
        estimator: graphlearn estimator wrapper
        group_min_size: int
        group_max_size int
        cluster_min_members:int
        cluster_max_members:int
        group_score_threshold: float
        group_score_classifier: KMeans(n_clusters=4)
        debug: bool
        subgraph_name_estimator: MiniBatchKMeans
        '''
        self.vectorizer=vectorizer
        self.estimator=estimator
        self.max_size=group_max_size
        self.min_size=group_min_size
        self.score_threshold=group_score_threshold
        self.debug=debug
        self.subgraph_name_estimator=subgraph_name_estimator
        self.save_graphclusters=save_graphclusters
        self.cluster_min_members=cluster_min_members
        self.cluster_max_members=cluster_max_members


    def fit_param_init(self):
        self.ignore_clusters=[]

    def fit(self,graphs):

        self.fit_param_init()

        # graphs will be used more than once, so if its a generator we want a list.
        graphs=list(graphs)

        # learning how to score nodes
        self.estimator.fit(self.vectorizer.transform(graphs))



        # a functon to generate a minorgraph, that contracts all groups
        self.abstractor = GraphToAbstractTransformer(
                score_threshold=self.score_threshold,
                min_size=self.min_size,
                max_size=self.max_size,
                debug=self.debug,
                estimator=self.estimator)

        #  groups will be clustered.
        subgraphs = self.abstractor.get_subgraphs(graphs)

        # we may need theese later.. and since it is an iterator...
        if self.save_graphclusters:
            subgraphs= list(subgraphs)

        data= self.vectorizer.transform( subgraphs )
        self.subgraph_name_estimator.fit(data)
        cluster_ids = self.subgraph_name_estimator.predict(data)


        for i in range(self.subgraph_name_estimator.get_params()['n_clusters']):
            cids=cluster_ids.tolist()
            members = cids.count(i)
            if members< self.cluster_min_members or members >  self.cluster_max_members > -1: # should work to omou
                logger.debug('remove cluser: %d  members: %d' % (i,members))
                self.ignore_clusters.append(i)



        # some information:
        logger.debug('num clusters: %d' % max(cluster_ids))
        logger.debug(report_base_statistics(cluster_ids).replace('\t', '\n'))
        if self.save_graphclusters:
            self.graphclusters = defaultdict(list)
            for cluster_id, graph in izip(cluster_ids, subgraphs):
                self.graphclusters[cluster_id].append(graph)



    def transform(self,graphs):
        '''

        Parameters
        ----------
        inputs: [graph]

        Returns
        -------
            [(edge_expanded_graph, minor),...]
        '''

        return [self.re_transform_single(graph) for graph in graphs]

    def re_transform_single(self, graph):
        '''
        Parameters
        ----------
        graph

        Returns
        -------
        a postprocessed graphwrapper
        '''
        return (edengraphtools._edge_to_vertex_transform(graph),
                rename_subgraph(graph,
                               self.abstractor,
                               self.subgraph_name_estimator,
                               self.vectorizer, self.ignore_clusters))








class GraphToAbstractTransformer(object):
    '''
    MAKE MINOR GRAPH LAYOUT

    makes abstractions that are based on the score of an estimator
    this class is just a helper for minor transform.
    '''

    def __init__(self, estimator=None, score_threshold=0.0, min_size=0,max_size=50, debug=False):
        '''

        Parameters
        ----------
        vectorizer  eden.graph.vectorizer
        estimator   estimator to assign scores

        score_threshold
            ignore nodes with score < thresh
        min_size
            min size for clusters
        debug
            debug mode?

        Returns
        -------

        '''
        self.vectorizer = Vectorizer()

        self.estimator = estimator
        self.score_threshold = score_threshold
        self.min_size = min_size
        self.debug = debug
        self.max_size=max_size


        class groupByMinScore:
            def __init__(self, score):
                self.min_score = score
            def fit(self, li):
                pass
            def predict(self, i):
                return [1 if i >= self.min_score else 0]
        self.grouper = groupByMinScore(self.score_threshold)

    """
    def set_parmas(self,**kwargs):
        '''

        Parameters
        ----------
        kwargs:
            vectorizer = a vectorizer to edge_vertex transform
            estimator = estimator object to assign scores to nodes
            grouper =  object with predict(score) function to assign clusterid to nodes
        Returns
        -------
        '''
        self.__dict__.update(kwargs)
    """
    def get_subgraphs(self, inputs, score_attribute='importance', group='class'):

        for graph in inputs:
            abstr = self._transform_single(graph, score_attribute=score_attribute, group=group)
            if self.debug:
                draw.graphlearn(abstr)
            for n, d in abstr.nodes(data=True):
                if len(d['contracted']) > 1 and 'edge' not in d and d.get('APPROVEDABSTRACTNODE', True):
                    # get the subgraph induced by it (if it is not trivial)
                    yield graph.subgraph(d['contracted']).copy()


    def _transform_single(self, graph, score_attribute='importance', group='class'):
        '''
        Parameters
        ----------
        score_attribute: string
            name of the attribute used
        group: string
            annnotate in this field
        Returns
        -------
        '''

        # graph expanded and unexpanded
        graph_exp = edengraphtools._edge_to_vertex_transform(graph)
        graph_unexp = edengraphtools._revert_edge_to_vertex_transform(graph_exp)

        # annotate with scores, then transform the score
        graph_unexp = self.vectorizer.annotate([graph_unexp], estimator=self.estimator.estimator).next()
        for n, d in graph_unexp.nodes(data=True):
            if d[score_attribute] > self.score_threshold:
                d[group] = str(self.grouper.predict(d[score_attribute])[0])
            else:
                d[group] = "-"

        if self.debug:
            print "##################################"
            print 'score annotation'
            for n,d in graph_unexp.nodes(data=True):
                d[score_attribute]=round(d[score_attribute],1)
            draw.graphlearn(graph_unexp, vertex_label = group, secondary_vertex_label=score_attribute,size=10)

        # weed out groups that are too small
        # assign_values_to_nodelabel(graph_unexp, group)
        graph3 = contraction([graph_unexp], contraction_attribute=group, modifiers=[], nesting=False,
                             dont_contract_attribute_symbol='-').next()
        for n, d in graph3.nodes(data=True):
            if len(d['contracted']) < self.min_size:
                for n in d['contracted']:
                    graph_unexp.node[n].pop(group)
                    graph_unexp.node[n][group] = '-'

            if len(d['contracted']) > self.max_size:
                scores= [ ( graph_unexp.node[n][score_attribute], n) for n in d['contracted'] ]
                scores.sort(reverse=True)
                #print 'scores',scores
                copygraph = graph_unexp.subgraph(d['contracted']).copy()

                def testsize(graph,maxsize,original, minsize):
                    ret=True
                    for g in nx.connected_component_subgraphs(graph):
                        if len(g) > maxsize:
                            ret = False
                        if len(g) < minsize:  # deleting things that become too small
                             for n in g.nodes():
                                 graph_unexp.node[n].pop(group)
                                 graph_unexp.node[n][group] = '-'
                                 # need to restore label i think

                    return ret

                while testsize(copygraph,self.max_size,graph_unexp,self.min_size)==False:
                    delnode=scores.pop()[1] # should give id of node with lowest score
                    #print 'deleting a node',delnode
                    copygraph.remove_node(delnode)
                    graph_unexp.node[delnode].pop(group)
                    graph_unexp.node[delnode][group] = '-'

        if self.debug:
            print 'checking group size constraint'
            print '[contraction, graph that should not contain groups that are too small]'
            draw.graphlearn([graph3, graph_unexp], vertex_label=group)


        # doing the real contraction
        graph_unexp = contraction([graph_unexp],
                                  contraction_attribute=group,
                                  modifiers=[],
                                  nesting=False,dont_contract_attribute_symbol='-').next()

        for n, d in graph_unexp.nodes(data=True):
            if d[group] == '-':
                d['APPROVEDABSTRACTNODE'] = False
        if self.debug:
            print 'final contraction:'
            draw.graphlearn(graph_unexp, vertex_label=group)



        # expand
        graph_reexp = edengraphtools._edge_to_vertex_transform(graph_unexp)
        #  make a dictionary that maps from base_graph_node -> node in contracted graph
        getabstr = {contra: node for node, d in graph_reexp.nodes(data=True) for contra in d.get('contracted', [])}
        # so this basically assigns edges in the base_graph to nodes in the abstract graph.
        for n, d in graph_exp.nodes(data=True):
            if 'edge' in d:
                # if we have found an edge node...
                # lets see whos left and right of it:
                n1, n2 = graph_exp.neighbors(n)
                # case1: ok those belong to the same gang so we most likely also belong there.
                if getabstr[n1] == getabstr[n2]:
                    graph_reexp.node[getabstr[n1]]['contracted'].add(n)

                # case2: neighbors belong to different gangs...
                else:
                    blub = set(graph_reexp.neighbors(getabstr[n1])) & set(graph_reexp.neighbors(getabstr[n2]))
                    for blob in blub:
                        if 'contracted' in graph_reexp.node[blob]:
                            graph_reexp.node[blob]['contracted'].add(n)
                        else:
                            graph_reexp.node[blob]['contracted'] = set([n])



        return graph_reexp

    def transform(self, graphs, score_attribute='importance', group='class', debug=False):
        for graph in graphs:
            yield self._transform_single(graph, score_attribute=score_attribute, group=group, debug=debug)






def rename_subgraph(graph, minorgenerator, nameestimator,vectorizer,ignore_labels):
    """
    relabels subgraphs by nameestimator

    Parameters
    ----------
    graph
    minorgenerator
    nameestimator
    vectorizer

    Returns
    -------
        relabeled graph
    """

    abst = minorgenerator._transform_single(graph, score_attribute='importance', group='class')
    subgraphs, ids = get_subraphs(abst,graph,minor_ids=True)
    if len(subgraphs)==0:
        return abst
    vectors = vectorizer.transform(subgraphs)
    clusterids = nameestimator.predict(vectors) # hope this works
    return set_labels(graph=abst,names=clusterids,ids=ids,labelprefix='C_', label="label", ignore_labels=ignore_labels)


def set_labels(graph,names,ids,labelprefix='', label="label",ignore_labels=[]):
    for name, id in zip(names,ids):
        if name not in ignore_labels:
            #print id, ignore_labels
            graph.node[id][label]=labelprefix+str(name)
        else:
            graph.node[id][label] = '-'

    return graph

def get_subraphs(minorgraph,graph,minor_ids=True):
    '''

    Parameters
    ----------
    minorgraph
    graph
    minor_ids, bool
        will also return a list of ids for the subgraphs

    Returns
    -------
        subgraphlist or (subgrpahlist,id_list)
    '''
    graph = edengraphtools._revert_edge_to_vertex_transform(graph)

    graphs=[]
    ids=[]
    for n, d in minorgraph.nodes(data=True):
        if len(d['contracted']) > 1 and 'edge' not in d:
            graphs.append(  graph.subgraph(d['contracted']).copy() )
            ids.append(n)
    if minor_ids:
        return graphs, ids

    return graphs


def get_all_scores(graphs, vectorizer, estimator):
    '''
    extracts all node scores

    Parameters
    ----------
    graphs: [graph]

    Returns
    -------
    '''
    li = []
    for graph in graphs:
        g = vectorizer.annotate([graph], estimator=estimator.estimator).next()
        for n, d in g.nodes(data=True):
            li.append([d['importance']])
    return li




















"""
class GraphMinorTransformer(GraphTransformer):
    def __init__(self,
                 node_name_grouper=KMeans(n_clusters=4),
                 name_cluster=MiniBatchKMeans(n_clusters=5),
                 save_graphclusters=False,
                 # graph_to_minor=GraphToAbstractTransformer(),
                 estimator=OneClassEstimator(nu=.5, n_jobs=4),
                 group_min_size=2,
                 group_max_size=5,
                 group_score_threshold=0,
                 debug=False):
        '''

        Parameters
        ----------
        node_name_grouper: KMeans()
            fittable cluster algo that clusters estimator scores
            you may also just use raw scores which works best with the shape_* parameters

        name_cluster: MiniBatchKMeans()
            fitable cluster algo that will run on core_shape_clusters

        save_graphclusters:
            saving the extracted core_shape_clusters
            ans order them by their name_cluster
        estimator
            oneclass estimator that will work on vectorized whole graphs

        group_min_size:
            influencing how a core(minor node) may look like, here we set a minimum size for the code
        group_score_threshold:
            influencing how a core(minor node) may look like,
            here we set a minimum score for the core.
        debug: False
            will print LOTS of graphs, so dont use this.

        Returns
        -------

        '''
        self.save_graphclusters = save_graphclusters
        self.name_cluster = name_cluster
        self.node_name_grouper = node_name_grouper
        self.rawgraph_estimator = estimator
        self.shape_score_threshold = group_score_threshold
        self.group_min_size = group_min_size
        self.group_max_size = group_max_size
        self.vectorizer = Vectorizer()
        self.debug=debug





    def fit(self, inputs):
        '''
        Parameters
        ----------
        inputs many nx.graph
        Returns
        -------
        '''
        inputs = list(inputs)
        # this k means is over the values resulting from annotation
        # and determine how a graph will be split intro minor nodes.
        vectorized_inputs = self.vectorizer.transform(inputs)
        self.rawgraph_estimator.fit(vectorized_inputs)

        # rawesti will write numbers to nodes, coreshapecluster renames these
        if 'min_score' not in self.node_name_grouper.__dict__:
            self.train_core_shape_cluster(inputs)

        # with these renamings we can create an abstract graph
        self._abstract = GraphToAbstractTransformer(score_threshold=self.shape_score_threshold,
                                                    min_size=self.group_min_size,
                                                    max_size=self.group_max_size,
                                                    debug=self.debug,
                                                    estimator=self.rawgraph_estimator,
                                                    grouper=self.node_name_grouper)

        # now comes the second part in which i try to find a name for those minor nodes.
        if self.name_cluster:
            self.train_name_cluster(inputs)

    def train_name_cluster(self, inputs):
        parts = []
        if len(inputs)==0:
            print 'lost input'
        # for all minor nodes:
        for graph in inputs:
            abstr = self._abstract._transform_single(graph, score_attribute='importance', group='class')
            for n, d in abstr.nodes(data=True):
                if len(d['contracted']) > 1 and 'edge' not in d and d.get('APPROVEDABSTRACTNODE', True):
                    # get the subgraph induced by it (if it is not trivial)
                    tmpgraph = graph.subgraph(d['contracted']).copy()
                    parts.append(tmpgraph)

        logger.debug("learning abstraction: %d partial graphs found" % len(parts))
        # draw.graphlearn(parts[:5], contract=False)
        # code from annotation-components.ipynb:
        data_matrix = self.vectorizer.transform(parts)

        self.name_cluster.fit(data_matrix)
        cluster_ids = self.name_cluster.predict(data_matrix)
        logger.debug('num clusters: %d' % max(cluster_ids))
        logger.debug(report_base_statistics(cluster_ids).replace('\t', '\n'))

        if self.save_graphclusters:
            self.graphclusters = defaultdict(list)
            for cluster_id, graph in izip(cluster_ids, parts):
                self.graphclusters[cluster_id].append(graph)

    def train_core_shape_cluster(self, inputs):
        '''
        extracts all node scores and clusters them.


        Parameters
        ----------
        inputs: [graph]

        Returns
        -------
        '''
        li = []
        for graph in inputs:
            g = self.vectorizer.annotate([graph], estimator=self.rawgraph_estimator.estimator).next()
            for n, d in g.nodes(data=True):
                li.append([d['importance']])

        self.node_name_grouper.fit(li)



    def fit_transform(self, inputs):
        '''
        Parameters
        ----------
        input : graphs

        Returns
        -------
        graphwrapper iterator
        '''

        inputs = list(inputs)
        self.fit(inputs)
        return self.transform(inputs)

    def re_transform_single(self, graph):
        '''
        Parameters
        ----------
        graph

        Returns
        -------
        a postprocessed graphwrapper
        '''

        # draw.graphlearn(graph)
        # print len(graph)
        # abstract = self.abstract(graph, debug=False)
        # draw.graphlearn([graph,abstract])
        return self.transform([graph])[0]

    def abstract(self, graph, score_attribute='importance', group='class', debug=False):
        '''
        Parameters
        ----------
        graph: nx.graph
        score_attribute: string, 'importance'
            attribute in which we write the score
        group: string, 'class'
            where to write clusterid
        debug: bool, False
            draw abstracted graphs

        Returns
        -------
            nx.graph: a graph minor
            if named_cluster is present, we the minor graph nodes will be named accordingly
        '''

        # generate abstract graph
        abst = self._abstract._transform_single(graph, score_attribute, group)
        if self.name_cluster == False:
            return abst

        graph = edengraphtools._revert_edge_to_vertex_transform(graph)
        for n, d in abst.nodes(data=True):
            if len(d['contracted']) >= self.group_min_size and 'edge' not in d:
                # get the subgraph induced by it (if it is not trivial)
                tmpgraph = graph.subgraph(d['contracted']).copy()
                vector = self.vectorizer.transform_single(tmpgraph)
                d['label'] = "C_" + str(self.name_cluster.predict(vector))

            elif len(d['contracted']) < self.group_min_size and 'edge' not in d:
                # get the subgraph induced by it (if it is not trivial)
                d['label'] = graph.node[list(d['contracted'])[0]]['label']


            elif 'edge' not in d:
                d['label'] = "F_should_not_happen"
        return abst

    def transform(self, inputs):
        '''

        Parameters
        ----------
        inputs: [graph]

        Returns
        -------
            [(edge_expanded_graph, minor),...]
        '''

        return [(edengraphtools._edge_to_vertex_transform(graph), self.abstract(graph)) for graph in inputs]
"""

