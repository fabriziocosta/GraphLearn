import networkx as nx
from eden import graph as edengraphtools
from eden.graph import Vectorizer
from eden_extra.modifier.graph.structure import contraction
from graphlearn01.utils import node_operation, remove_eden_annotation , draw
from sklearn.base import BaseEstimator, TransformerMixin
import graphlearn01.utils as utils
import logging
logger = logging.getLogger(__name__)

import graphlearn01.minor.decompose as decompose_minor
import graphlearn01.decompose as decompose # decompose .  extract core interface should be this
from collections import defaultdict

class RmTrash(object):

    def __init__(self,attribute=lambda x:x, threshold=.5, min_size=3,max_size=10):
        self.attribute = attribute
        self.threshold_low=threshold
        self.threshold_high=1-threshold
        self.min_size= min_size
        self.max_size= max_size


    def cut(self,graphs):
        for graph in graphs:
            graph=graph.copy()
            for n in graph.nodes():
                if  self.threshold_low < self.attribute(graph.node[n]) < self.threshold_high:
                    graph.remove_node(n)
            for sg in nx.connected_component_subgraphs(graph):
                if self.min_size <= len(sg) <= self.max_size:
                    yield sg



class Cutter(object):

    def __init__(self,attribute=lambda x:x, threshold=.5, min_size=3,max_size=10):
        self.attribute = attribute
        self.threshold=threshold
        self.min_size= min_size
        self.max_size= max_size

    def cut_single(self,graph):
        graph=graph.copy()
        #asd = self.attribute
        #self.attribute = lambda x:x['tmpscore']

        for a,b in graph.edges():
            if  abs(self.attribute(graph.node[a]) - self.attribute(graph.node[b])) > self.threshold:
                graph.remove_edge(a,b)
        for sg in nx.connected_component_subgraphs(graph):
            if self.min_size <= len(sg) <= self.max_size:
                yield sg


    def cut(self,graphs):
        for graph in graphs:
            for e in self.cut_single(graph):
                yield e







def merge(graph,core):
    nodes=core.nodes()
    for node_id in nodes[1:]:
        for n in graph.neighbors(node_id):
            graph.add_edge(nodes[0], n, graph[node_id][n])
        graph.remove_node(node_id)

    graph.remove_edge(nodes[0],nodes[0])
    return nodes[0]




class cutter_with_interface(Cutter):
    def transform2(self,graphs, thickness=2):

        for graph in graphs:
            cc = list(self.cut_single(graph))
            for core in cc:
                g=graph.copy()
                root = merge(g,core)

                #cip = decompose_minor.extract_core_and_interface(root, g, radius_list=[0], thickness_list=[thickness])

                cip = decompose_minor.extract_cips(root, deci(graph, g, root, core.nodes()),
                                                   base_thickness_list=[thickness],
                                                   hash_bitmask=2 * 20 - 1,
                                                   radius_list = [0], thickness_list = [2])
                if len(cip) == 0:
                    continue

                #core.graph['interface_hash']=cip[0].interface_hash
                yield core,cip[0]


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
        #"""Transform."""
        #try:
        self.counter = 0
        for graph in graphs:
            ccomponents = self._extract_ccomponents(
                graph.copy(),
                threshold=self.threshold,
                min_size=self.min_size,
                max_size=self.max_size)
            yield ccomponents
        pass

        #except Exception as e:
        #    print ('Failed iteration. Reason: %s' % e)


    def get_attr_from_noded(self,d):
        return self.attribute(d)
        #return d.get(self.attribute,[False,False])[0]


    def _extract_ccomponents(self, graph, threshold=0, min_size=2, max_size=20):
        # remove all vertices that have a score less then threshold
        cc_list = []

        if self.less_then:
            less_component_graph = graph.copy()
            for v, d in less_component_graph.nodes_iter(data=True):
                if self.get_attr_from_noded(d):
                    if self.get_attr_from_noded(d) < threshold:
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
                if self.get_attr_from_noded(d):
                    if self.get_attr_from_noded(d) >= threshold:
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
            scores = [(self.get_attr_from_noded(d), n) for n, d in graph.nodes(data=True)]
            graph.remove_node(choose_cut_node(scores)[1])
            # check the resulting components
            for g in nx.connected_component_subgraphs(graph):
                if len(g) > max_size:
                    checklist.append(g)
                elif len(g) >= min_size:
                    yield g


def deci(big, small, root, merged):
        b = big.copy()
        s = small
        # we need r and m to set up the contracted sets...
        for n,d in s.nodes(data=True):
            d['contracted'] = set([n])
            if n==root:
                for asd in merged:
                    d['contracted'].add(asd)

        #s= nx.convert_node_labels_to_integers(s, max(b.nodes())+1)

        s.graph['original']=b

        # calc_contracted_edge_nodes=True,  there is this option for the minordecomp.. dunno if need
        edengraphtools._label_preprocessing(s)
        edengraphtools._label_preprocessing(b)
        return decompose_minor.MinorDecomposer(s)



class TCC_with_interface(ThresholdedConnectedComponents):
    def transform2(self,graphs, thickness=2):
        for graph,cc in zip(graphs, self.transform(graphs)):
            for core in cc:
                g=graph.copy()
                root = merge(g,core)
                if False:
                    import structout as so
                    def pg(g):
                        for n,d in g.nodes(data=True):
                            d['id']=str(n)
                        return g
                    thing = deci(graph, g, root, core.nodes())
                    so.gprint(pg(g), size=30, label='contracted')
                    so.gprint(pg(graph), size=30, label='id')
                    so.gprint(pg(thing._base_graph), size=30, label='id')
                    so.gprint(pg(thing._abstract_graph),size=30, label='id')
                    for n,d in thing._abstract_graph.nodes(data=True):
                        print d
                cip = decompose_minor.extract_cips(root, deci(graph, g, root, core.nodes()),
                                                   base_thickness_list=[thickness],
                                                   hash_bitmask=2 * 20 - 1,
                                                   radius_list=[0], thickness_list=[2])
                #cip = decompose.extract_core_and_interface(root, g, radius_list=[0],
                #                                           thickness_list=[thickness] )
                if len(cip) == 0:
                    continue
                #core.graph['interface_hash']=cip[0].interface_hash
                yield core,cip[0]


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
                 subgraphextraction='best',
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
        self.subgraphextraction=subgraphextraction

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

        cutter=Cutter(
                attribute= lambda d: d.get( self.score_attribute,[False])[0],
                max_size=self.max_size,
                min_size=self.min_size,
                threshold=self.score_threshold
                )
        return list(cutter.cut(inputs))

        '''
        def add_ihash(core,cip):
            core.graph['interface_hash']=cip.interface_hash

            return core

        def base_cip(core,cip):

            #draw.graphlearn(cip.graph)
            try:
                res=  edengraphtools._revert_edge_to_vertex_transform(cip.graph)
            except:
                print 'abstractor.py failed to return base cip...'
                import structout as so
                so.gprint(cip.graph)

            return res

        if self.subgraphextraction == 'best_interface':
            tcc = TCC_with_interface(attribute=lambda d: d.get( self.score_attribute,[False])[0],
                                      more_than=False,less_then=True, shrink_graphs=True,
                                             threshold=self.score_threshold,
                                             min_size=self.min_size,
                                             max_size=self.max_size)

            return [ add_ihash(a,b) for (a,b) in tcc.transform2(inputs, thickness=2) ]


        if self.subgraphextraction == 'best_soft_interface':
            tcc = TCC_with_interface(attribute=lambda d: d.get( self.score_attribute,[False])[0],
                                      more_than=False,less_then=True, shrink_graphs=True,
                                             threshold=self.score_threshold,
                                             min_size=self.min_size,
                                             max_size=self.max_size)

            res =  [ base_cip(a,b) for (a,b) in tcc.transform2(inputs, thickness=2) ]
            return res

        if self.subgraphextraction == 'best':
            res=  list(standalone_get_subgraphs(inputs,
                                                lambda d: d.get( self.score_attribute,[False])[0],
                                                self.group_attribute,
                                                self.score_threshold,
                                                self.min_size,
                                                self.max_size))
            return res


        if self.subgraphextraction == 'cut':
                cutter=Cutter(
                    attribute= lambda d: d.get( self.score_attribute,[False])[0],
                    max_size=self.max_size,
                    min_size=self.min_size,
                    threshold=self.score_threshold)
                return list(cutter.cut(inputs))


        if self.subgraphextraction == 'cut_soft_interface':
            cutter=cutter_with_interface(
                attribute= lambda d: d.get( self.score_attribute,[False])[0],
                max_size=self.max_size,
                min_size=self.min_size,
                threshold=self.score_threshold)

            res= [ base_cip(a,b) for (a,b) in cutter.transform2(inputs, thickness=2) ]
            return res

        if self.subgraphextraction == 'cut_interface':
            cutter=cutter_with_interface(
                attribute= lambda d: d.get( self.score_attribute,[False])[0],
                max_size=self.max_size,
                min_size=self.min_size,
                threshold=self.score_threshold)
            return [ add_ihash(a,b) for (a,b) in cutter.transform2(inputs) ]



        print "please tell me the subgraphextraction strategy"

        '''
        res +=  list(standalone_get_subgraphs(inputs,                         # adding negatives
                                            lambda d: d.get(self.score_attribute,[False,False])[1],
                                            self.group_attribute,
                                            self.score_threshold,
                                            self.min_size,
                                            self.max_size,more_then=True,less_then=False))
        '''
        #while not res:
        #    print "OHSNAP"
        #    self.score_threshold*=.9
        #    res=  list(standalone_get_subgraphs(inputs, self.score_attribute, self.group_attribute, self.score_threshold, self.min_size,
        #                               self.max_size))
        return res




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

        #tcc = ThresholdedConnectedComponents(attribute=  lambda d: d.get( self.score_attribute,[False])[0] , more_than=False, shrink_graphs=True)
        try:
            components=self.get_subgraphs([graph])


        except Exception as inst:
            s= 'abstractor.py Thresholdedconnectedcomponents failed\n'
            for node,d in graph.nodes(data=True):
                s+=str(node)+str(d)+"\n"
            logger.log(20,s)
            raise inst

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
            print 'name_estimation learnedlayer abstractor, sumsubgraphs: %d, draw:' % len(subgraphs)
            #draw.graphlearn(subgraphs, contract= False)
            for e in subgraphs:
                print utils.ascii.nx_to_ascii(e)
                import pprint
                pprint.pprint( e.nodes(data=True))
                #draw.debug(e)

        clusterids = nameestimator.predict(data,subgraphs)

        #for d, g in zip(data, subgraphs):
        #    g.graph['hash_title'] = hash_function(d)
        #draw.graphlearn(subgraphs,size=2, title_key='hash_title', edge_label='label')

        for sg, clid in zip(subgraphs, clusterids):
            try:
                for n in sg.nodes():
                    graph.node[n][group] = '-' if clid == -1 else str(clid)
            except:
                import structout as so
                so.gprint(graph, label='id')
                so.gprint(sg, label='id')

    # doing the contraction...
    graph = contraction([graph], contraction_attribute=group, modifiers=[],
                        nesting=False, dont_contract_attribute_symbol='-').next()

    # write labels
    def f(n, d):
        d['label'] = graphreference.node[max(d['contracted'])]['label'] \
            if d['label'] == '-' else "L%sC%s" % (layer, d['label'])
        d.pop("importance",None)

    node_operation(graph, f)
    return graph



def standalone_get_subgraphs(inputs, nodedict_to_score, group, threshold, min_size, max_size,more_then=False, less_then=True):
    tcc = ThresholdedConnectedComponents(attribute=nodedict_to_score, more_than=more_then,less_then=less_then, shrink_graphs=True,
                                         threshold=threshold,
                                         min_size=min_size,
                                         max_size=max_size)
    for lyst in tcc.transform(inputs):
        for e in lyst:
            yield e





