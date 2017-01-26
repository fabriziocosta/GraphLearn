import networkx as nx
from eden.graph import Vectorizer
from eden_extra.modifier.graph.structure import contraction
from graphlearn.utils import node_operation, remove_eden_annotation , draw
from sklearn.base import BaseEstimator, TransformerMixin

class ThresholdedConnectedComponents(BaseEstimator, TransformerMixin):
    """ThresholdedConnectedComponents."""

    def __init__(self, attribute='importance', threshold=0, min_size=3, max_size=20,
                 shrink_graphs=False,
                 less_then=True, more_than=True):
        """Construct."""
        self.attribute = attribute
        self.threshold = threshold
        self.min_size = min_size
        self.less_then = less_then
        self.more_than = more_than
        self.max_size = max_size
        self.shrink_graphs = shrink_graphs
        self.counter = 0  # this guy looks like hes doing nothing?

    def transform(self, graphs):
        """Transform."""
        try:
            self.counter = 0
            for graph in graphs:
                ccomponents = self._extract_ccomponents(
                    graph,
                    threshold=self.threshold,
                    min_size=self.min_size,
                    max_size=self.max_size)
                yield ccomponents
            pass
        except Exception as e:
            print ('Failed iteration. Reason: %s' % e)
            #print ('Exception', exc_info=True)

    def _extract_ccomponents(self, graph, threshold=0, min_size=2, max_size=20):
        # remove all vertices that have a score less then threshold
        cc_list = []

        if self.less_then:
            less_component_graph = graph.copy()
            for v, d in less_component_graph.nodes_iter(data=True):
                if d.get(self.attribute, False):
                    if d[self.attribute] < threshold:
                        less_component_graph.remove_node(v)
            for cc in nx.connected_component_subgraphs(less_component_graph):
                if len(cc) >= min_size and len(cc) <= max_size:
                    cc_list.append(cc)
                if len(cc) > max_size and self.shrink_graphs:
                    cc_list += list(self.enforce_max_size(cc, min_size, max_size))

        # remove all vertices that have a score more then threshold
        if self.more_than:
            more_component_graph = graph.copy()
            for v, d in more_component_graph.nodes_iter(data=True):
                if d.get(self.attribute, False):
                    if d[self.attribute] >= threshold:
                        more_component_graph.remove_node(v)

            for cc in nx.connected_component_subgraphs(more_component_graph):
                if len(cc) >= min_size and len(cc) <= max_size:
                    cc_list.append(cc)

                if len(cc) > max_size and self.shrink_graphs:
                    cc_list += list(self.enforce_max_size(cc, min_size, max_size, choose_cut_node=max))

        return cc_list

    def enforce_max_size(self, graph, min_size, max_size, choose_cut_node=min):
        # checklist contains graphs that are too large.
        checklist = [graph]
        while checklist:
            # remove lowest scoring node:
            graph = checklist.pop()
            scores = [(d[self.attribute], n) for n, d in graph.nodes(data=True)]
            graph.remove_node(choose_cut_node(scores)[1])
            # check the resulting components
            for g in nx.connected_component_subgraphs(graph):
                if len(g) > max_size:
                    checklist.append(g)
                elif len(g) >= min_size:
                    yield g


class GraphToAbstractTransformer(object):
    '''
    MAKE MINOR GRAPH LAYOUT
    makes abstractions that are based on the score of an estimator
    this class is just a helper for minor transform.
    '''

    def __init__(self, estimator=None,
                 score_threshold=0.0,
                 min_size=0,
                 max_size=50,
                 debug=False,
                 layer=False,
                 vectorizer=Vectorizer(),
                 score_attribute='importance',
                 group_attribute ='class'):
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
        self.max_size = max_size
        self.layer = layer
        self.score_attribute=score_attribute
        self.group_attribute = group_attribute

    def get_subgraphs(self, inputs):
        '''
        calls get_subgraph_single on all graphs, has option to go multiprocess

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

        return standalone_get_subgraphs(inputs, self.score_attribute, self.group_attribute, self.score_threshold, self.min_size,
                                        self.max_size)


    def transform(self,graphs):
        return [self._transform_single(g) for g in graphs]

    def _transform_single(self, graph):
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

        graphcopy = graph.copy()
        maxnodeid = max(graph.nodes())

        # def f(n,d): d[score_attribute] = graph.degree(n)
        # node_operation(graph,f)

        tcc = ThresholdedConnectedComponents(attribute=self.score_attribute, more_than=False, shrink_graphs=True)
        components = tcc._extract_ccomponents(graph, threshold=self.score_threshold, min_size=self.min_size,
                                              max_size=self.max_size)

        nodeset = {n for g in components for n in g.nodes()}

        def f(n, d):
            d[self.group_attribute] = '~' if n in nodeset else '-'

        node_operation(graph, f)

        # now we either contract what we have, or additionally rename the contracted nodes according to the group estimator

        graph = name_estimation(graph, self.group_attribute, self.layer, graphcopy, self.vectorizer, self.nameestimator,
                                    components)
        #else:
        #    graph = contraction([graph],
        #                        contraction_attribute=group,
        #                        modifiers=[],
        #                        nesting=False, dont_contract_attribute_symbol='-').next()

        graph = nx.relabel_nodes(graph, dict(
            zip(graph.nodes(), range(maxnodeid + 1, 1 + maxnodeid + graph.number_of_nodes()))), copy=False)

        graph.graph['original']= graphcopy
        graph.graph['layer']=self.layer
        return graph


def name_estimation(graph, group, layer, graphreference, vectorizer, nameestimator, subgraphs):
    if subgraphs:
        map(remove_eden_annotation, subgraphs)
        try:
            data = vectorizer.transform(subgraphs)
        except:
            draw.graphlearn(subgraphs, contract= False)
        clusterids = nameestimator.predict(data)

        #for d, g in zip(data, subgraphs):
        #    g.graph['hash_title'] = hash_function(d)
        #draw.graphlearn(subgraphs,size=2, title_key='hash_title', edge_label='label')

        for sg, clid in zip(subgraphs, clusterids):
            for n in sg.nodes():
                graph.node[n][group] = '-' if clid == -1 else str(clid)

    # doing the contraction...
    graph = contraction([graph], contraction_attribute=group, modifiers=[],
                        nesting=False, dont_contract_attribute_symbol='-').next()

    # write labels
    def f(n, d):
        d['label'] = graphreference.node[max(d['contracted'])]['label'] \
            if d['label'] == '-' else "L%sC%s" % (layer, d['label'])

    node_operation(graph, f)
    return graph



def standalone_get_subgraphs(inputs, score_attribute, group, threshold, min_size, max_size):
    tcc = ThresholdedConnectedComponents(attribute=score_attribute, more_than=False, shrink_graphs=True,
                                         threshold=threshold,
                                         min_size=min_size,
                                         max_size=max_size)
    for lyst in tcc.transform(inputs):
        for e in lyst:
            yield e





