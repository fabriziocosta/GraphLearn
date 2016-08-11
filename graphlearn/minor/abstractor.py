from eden.modifier.graph.structure import contraction
from eden.graph import Vectorizer
import networkx as nx
from graphlearn.utils import draw


def get_subgraphs_single(minorgraph, graph, minor_ids=False):
    '''
    Parameters
    ----------
    minorgraph
    graph
    minor_ids, bool

    Returns
    -------
        subgraphlist or (subgrpahlist,
                        ids_of_minor corresponding to subgraphlist,
                        ids_of_minor not producing subgraphs)
    '''
    graphs = []
    grouped_ids = []
    ungrouped_ids = []
    for n, d in minorgraph.nodes(data=True):
        if len(d['contracted']) > 1 and 'edge' not in d:
            graphs.append(graph.subgraph(d['contracted']).copy())
            grouped_ids.append(n)
        else:
            ungrouped_ids.append(n)
    if minor_ids:
        return graphs, grouped_ids, ungrouped_ids
    return graphs



def node_operation(graph, f):
    # applies function to n,d of nodes(data=True)
    # if you want to do assignments do this: def my(n,d): d[dasd]=asd
    # if you want the result you may use lambda :)
    res=[]
    for n,d in graph.nodes(data=True):
        res.append(f(n,d))
    return res


def apply_size_constrains(graph, min_size, max_size, group_attribute,score_attribute):
    '''
    graph, score and group annotated oO


    look at subgraphs induced by adjacent nodes whose group attribute is the same but not '-'.
    if the size of this construct is not in min and max size, group_attribute is set to '-'.

    Returns
    -------
        nothing, manipulates the group_attribute fields
    '''
    # contract on the new '~' and '-' annotations :)
    graph3 = contraction([graph], contraction_attribute=group_attribute, modifiers=[], nesting=False,
                         dont_contract_attribute_symbol='-').next()

    for n, d in graph3.nodes(data=True):
        # too small?
        if len(d['contracted']) < min_size:
            for n in d['contracted']:
                graph.node[n][group_attribute] = '-'

        # too big?
        if len(d['contracted']) > max_size:

            scores = [ (graph.node[n][score_attribute], n) for n in d['contracted'] ]
            scores.sort(reverse=True)
            copygraph = graph.subgraph(d['contracted']).copy()


            # if too big and split up, there might be groups that are too small
            def testsize(graph, maxsize, original, minsize):

                ret = True
                for g in nx.connected_component_subgraphs(graph):
                    if len(g) > maxsize:
                        ret = False
                    if len(g) < minsize:  # deleting things that become too small
                        for n in g.nodes():
                            original.node[n][group_attribute] = '-'
                            # need to restore label i think dsd
                return ret


            while testsize(copygraph, max_size, graph, min_size) == False:
                delnode = scores.pop()[1]  # should give id of node with lowest score
                # print 'deleting a node',delnode
                copygraph.remove_node(delnode)
                graph.node[delnode][group_attribute] = '-'


def name_estimation(graph, group,layer,graphreference, vectorizer, nameestimator):

    #draw.graphlearn(graph.copy(), secondary_vertex_label=group, size=10)

    # find labels and make sure that subgraphs with the -1 label dont get contracted
    graph3 = contraction([graph], contraction_attribute=group, modifiers=[], nesting=False,
                         dont_contract_attribute_symbol='-').next()

    subgraphs, grouped_ids, ungrouped_ids = get_subgraphs_single(graph3, graph, minor_ids=True)
    if len(subgraphs) != 0:
        vectors = vectorizer.transform(subgraphs)
        clusterids = nameestimator.predict(vectors)
        for i, clusterid in enumerate(clusterids):
            nodes = graph3.node[grouped_ids[i]]['contracted']
            # write groups
            for n in nodes:
                graph.node[n][group] =  '-' if clusterid == -1 else str(clusterid)



    # doing the contraction
    graph = contraction([graph],
                        contraction_attribute=group,
                        modifiers=[],
                        nesting=False, dont_contract_attribute_symbol='-').next()

    # write labels
    for n, d in graph.nodes(data=True):
        # label is - -> use contracted
        if d['label'] == '-':
            d['label'] = graphreference.node[max(d['contracted'])]['label']
        # label is number -> something
        else:
            d['label'] = "L" + str(layer) + "C" + str(d['label'])
    return graph
    # maybe this to save somelines :)
    # def f(n,d): d['label'] =    graphreference.node[max(d['contracted'])]['label'] if d['label'] ==  '-' else "L_" + str(layer) + "_C_" + str(d['label'])
    # node_operation(graph, f)


class GraphToAbstractTransformer(object):
    '''
    MAKE MINOR GRAPH LAYOUT
    makes abstractions that are based on the score of an estimator
    this class is just a helper for minor transform.
    '''
    def __init__(self, estimator=None, score_threshold=0.0, min_size=0,max_size=50, debug=False,layer=False,vectorizer=Vectorizer()):
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
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.score_threshold = score_threshold
        self.min_size = min_size
        self.debug = debug
        self.max_size=max_size
        self.layer=layer



    def get_subgraphs(self, inputs, score_attribute='importance', group='class'):
        '''
        Parameters
        ----------
        inputs, graph iterable
        score_attribute, field to write the score to
            better ignore this atm
        group, graph attr to write the group in.
            better ignore this atm
        Returns
        -------
            Use estimator to annotate graph, group important nodes together to induce subgraphs.
            yields subgraphs
        '''
        for graph in inputs:
            abstr = self._transform_single(graph, score_attribute=score_attribute, group=group)
            for g in get_subgraphs_single(abstr, graph, minor_ids=False):
                yield g


    def _transform_single(self, graph, score_attribute='importance', group='class', apply_name_estimation=False):
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
        graphcopy= graph.copy()


        # annotate with scores, then transform the score
        graph = self.vectorizer.annotate([graph], estimator=self.estimator).next()

        #def f(n,d): d[score_attribute] = graph.degree(n)
        #node_operation(graph,f)



        # apply threshold: label scores "-" or "1"
        def f(n,d):d[group] = '~' if d[score_attribute] > self.score_threshold else '-'
        node_operation(graph,f)


        # controll size of contracted subgraphs
        apply_size_constrains(graph, self.min_size, self.max_size, group, score_attribute)


        # now we either contract what we have, or additionally rename the contracted nodes according to the group estimator
        if apply_name_estimation:
            graph= name_estimation(graph, group,self.layer,graphcopy, self.vectorizer, self.nameestimator)
        else:
            graph = contraction([graph],
                                      contraction_attribute=group,
                                      modifiers=[],
                                      nesting=False,dont_contract_attribute_symbol='-').next()
        return graph

