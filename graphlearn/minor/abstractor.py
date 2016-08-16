from eden.modifier.graph.structure import contraction
from eden.graph import Vectorizer
import networkx as nx
from graphlearn.utils import draw
import eden
import multiprocessing as mp
from GArDen.decompose import ThresholdedConnectedComponents
import copy


def node_operation(graph, f):
    # applies function to n,d of nodes(data=True)
    # if you want to do assignments do this: def my(n,d): d[dasd]=asd
    # if you want the result you may use lambda :)
    res = []
    for n, d in graph.nodes(data=True):
        res.append(f(n, d))
    return res


def name_estimation(graph, group, layer, graphreference, vectorizer, nameestimator):
    # find new labels according to nameestimator
    # thresholded components also works with bytes :D
    tcc = ThresholdedConnectedComponents(attribute=group, more_than=False, shrink_graphs=False)
    subgraphs = tcc._extract_ccomponents(graph, threshold='A', min_size=2, max_size=99)  # '-' < 'A' < '~'
    if subgraphs:
        clusterids = nameestimator.predict(vectorizer.transform(subgraphs))
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


def multi(instances, abstractor):
    return abstractor.get_subgraphs(instances)


class GraphToAbstractTransformer(object):
    '''
    MAKE MINOR GRAPH LAYOUT
    makes abstractions that are based on the score of an estimator
    this class is just a helper for minor transform.
    '''

    def __init__(self, estimator=None, score_threshold=0.0, min_size=0, max_size=50, debug=False, layer=False,
                 vectorizer=Vectorizer()):
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

    def get_subgraphs(self, inputs, score_attribute='importance', group='class', multi_process=False):
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

        if multi_process == False:
            res = []
            tcc = ThresholdedConnectedComponents(attribute=score_attribute, more_than=False, shrink_graphs=True)
            for graph in inputs:
                if graph:
                    self.vectorizer.annotate([graph], estimator=self.estimator).next()
                    subgraphs = tcc._extract_ccomponents(graph, threshold=self.score_threshold, min_size=self.min_size,
                                                         max_size=self.max_size)
                    res += subgraphs
            return res
        else:
            pool = mp.Pool()
            mpres = [eden.apply_async(pool, multi, args=(graphs, self)) for graphs in eden.grouper(inputs, 50)]
            result = []
            for res in mpres:
                result += res.get()
            pool.close()
            pool.join()
            return result

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

        graphcopy = graph.copy()
        maxnodeid = max(graph.nodes())

        # annotate with scores, then transform the score
        graph = self.vectorizer.annotate([graph], estimator=self.estimator).next()

        # def f(n,d): d[score_attribute] = graph.degree(n)
        # node_operation(graph,f)

        tcc = ThresholdedConnectedComponents(attribute=score_attribute, more_than=False, shrink_graphs=True)
        components = tcc._extract_ccomponents(graph, threshold=self.score_threshold, min_size=self.min_size,
                                              max_size=self.max_size)
        nodeset = {n for g in components for n in g.nodes()}

        def f(n, d): d[group] = '~' if n in nodeset else '-'
        node_operation(graph, f)

        # now we either contract what we have, or additionally rename the contracted nodes according to the group estimator
        if apply_name_estimation:
            graph = name_estimation(graph, group, self.layer, graphcopy, self.vectorizer, self.nameestimator)
        else:
            graph = contraction([graph],
                                contraction_attribute=group,
                                modifiers=[],
                                nesting=False, dont_contract_attribute_symbol='-').next()

        graph = nx.relabel_nodes(graph, dict(
            zip(graph.nodes(), range(maxnodeid + 1, 1 + maxnodeid + graph.number_of_nodes()))), copy=False)
        return graph
