'''
automatic minor graph generation


# it also seems that the cascade has this interface to GL:
    -- fit, transform, fit_transform, retransform

cascade  -- manages a stack of transformers, OMG I SHOULD RLY DEFINE AN INTERF
transform  --  "annotate" and "abstractor"
annotate  -- write score to nodes
abstractor -- uses name_subgraph, probably mainly extract subgraphs
name_subgraph -- gets a bunch of subgraphs and trains does the clustering


each transformer does this:
    train a classifier
    annotate
    extract
    train another classifier
    contract
    // also it seems that it handles pos/neg input... also it seems that retransform has to contract the graph


steps to debug would be:

train class
annotate + get subgrs
train classifier
abstract


'''
import eden
import transform
from name_subgraphs import ClusterClassifier_keepduplicates as CC_keep
from name_subgraphs import ClusterClassifier as CC_nokeep
from name_subgraphs import ClusterClassifier_keepduplicates_interfaced as CC_keep_interface
from name_subgraphs import ClusterClassifier_fake as CC_noclust
from name_subgraphs import ClusterClassifier_soft_interface as CC_soft_inter
import networkx as nx
from graphlearn01.utils import draw
import graphlearn01.utils as utils



class Cascade(object):

    def toggledebug(self):
        self.debug= not self.debug
        for e in self.transformers:
            e.toggledebug()


    def __init__(self, depth=2,
                 debug=False,
                 multiprocess=True,
                 max_group_size=6,
                 min_group_size=2,
                 group_score_threshold=0,
                 clusterclassifier='keep',
                 subgraphextraction='best', # cut , best_interface
                 num_classes=2,
                 min_clustersize=2,
                 dbscan_range=.5,
                 vectorizer_annotation=eden.graph.Vectorizer(complexity=3),
                 vectorizer_cluster=eden.graph.Vectorizer(complexity=3),
                 annotate_dilude_score=False,
                 interfaceweight=0,
                 debug_rna=False):


        if type(dbscan_range) != list:
            dbscan_range= [dbscan_range]*depth
        self.dbscan_range=dbscan_range
        if type(group_score_threshold) != list:
            group_score_threshold=[group_score_threshold]*depth
        self.group_score_threshold = group_score_threshold


        if clusterclassifier== 'keep':
            self.makeclusterclassifier = lambda **kwargs: CC_keep(**kwargs)

        elif clusterclassifier == 'nokeep':
            self.makeclusterclassifier = lambda **kwargs: CC_nokeep(**kwargs)

        elif clusterclassifier == 'interface_nocluster':
            self.makeclusterclassifier = lambda **kwargs: CC_noclust(**kwargs)

        elif clusterclassifier == 'interface_keep':
            self.makeclusterclassifier = lambda **kwargs: CC_keep_interface(**kwargs)

        elif clusterclassifier == 'soft':
            self.makeclusterclassifier = lambda **kwargs: CC_soft_inter(interfaceweight=interfaceweight,**kwargs)

        else:
            exit()

        self.subgraphextraction= subgraphextraction
        self.debug_rna=debug_rna
        self.min_clustersize=min_clustersize
        self.depth = depth
        self.debug = debug
        self.multiprocess = multiprocess
        self.max_group_size = max_group_size
        self.min_group_size = min_group_size
        self.num_classes= num_classes
        self.vectorizer_annotation=vectorizer_annotation
        self.vectorizer_cluster = vectorizer_cluster
        self.annotate_dilude_score=annotate_dilude_score



        if debug:
            print "an instance of cascade was created:"
            print "dbscan_range: ",dbscan_range
            print "min_clustersize: ",min_clustersize
            print "min_group_size: ",min_group_size
            print "vectorizer_cluster: ",vectorizer_cluster
            print ""



    def setup_transformers(self):
        self.transformers = []
        for i in range(self.depth):
            transformer = transform.GraphMinorTransformer(
                vectorizer=self.vectorizer_annotation,
                cluster_classifier= self.makeclusterclassifier(debug=self.debug,vectorizer=self.vectorizer_cluster, min_clustersize=self.min_clustersize,dbscan_range=self.dbscan_range[i]),
                num_classes=self.num_classes,
                group_score_threshold= self.group_score_threshold[i],
                group_max_size=self.max_group_size,
                group_min_size=self.min_group_size,
                multiprocess=self.multiprocess,
                annotate_dilude_score=self.annotate_dilude_score,
                subgraphextraction=self.subgraphextraction,
                # cluster_max_members=-1,
                layer=i,
                debug=self.debug,
                debug_rna=self.debug_rna)
            self.transformers.append(transformer)


    def fit_transform_2(self, graphs, graphs_neg=[],remove_intermediary_layers=True):
        """ this fit transform will sub divide the pos and neg graphs in these parts:
        1. fit annotator
        2. fit abstractor
        3. only do transform on the 3rd set.

        only uses transformer.fit2 instead of the default transformer.fit_transform
        """

        if self.depth==0:
            return map(add_fake_abstract_layer,list(graphs)+list(graphs_neg))


        # INIT
        graphs=list(graphs)
        graphs_neg=list(graphs_neg)
        self.setup_transformers()
        for g in graphs+graphs_neg:
            g.graph['layer']=0


        numpos=len(graphs)
        graphs+=graphs_neg
        # fitting
        for i in range(self.depth):
            graphs = self.transformers[i].fit_2(graphs[:numpos], graphs[numpos:],fit_transform=True)
        if remove_intermediary_layers:
            graphs = self.do_remove_intermediary_layers(graphs)

        if self.debug:
            print "cascase: full transformation"
            draw.graphlearn_layered2(graphs[:10], vertex_label='label',scoretricks=True, edge_label='label' )


        return graphs

    def fit_transform(self, graphs, graphs_neg=[],remove_intermediary_layers=True):
        if self.depth==0:
            return map(add_fake_abstract_layer,list(graphs)+list(graphs_neg))


        # INIT
        graphs=list(graphs)
        graphs_neg=list(graphs_neg)
        self.setup_transformers()
        for g in graphs+graphs_neg:
            g.graph['layer']=0


        numpos=len(graphs)
        graphs+=graphs_neg
        # fitting
        for i in range(self.depth):
            graphs = self.transformers[i].fit_transform(graphs[:numpos], graphs[numpos:])
        if remove_intermediary_layers:
            graphs = self.do_remove_intermediary_layers(graphs)

        if self.debug:
            #draw.graphlearn([graphs[0], graphs[0].graph['original']], contract =False, vertex_label='contracted')
            #for n , d in graphs[0].graph['original'].nodes(data=True):
            #    print n, d
            print "cascase: full transformation"

            #draw.graphlearn(graphs[:5], contract=False, size=7,vertex_size=600, vertex_label='importance',font_size=9,secondary_vertex_label='label',edge_label='label')
            draw.graphlearn_layered2(graphs[:10], vertex_label='label',scoretricks=True, edge_label='label' )
        return graphs




    def fit(self, graphs, g2=[]):
        self.fit_transform(graphs,g2)
        return self

    def transform(self, graphs, remove_intermediary_layers=True):
        if self.depth==0:
            return map(add_fake_abstract_layer,graphs)

        graphs = map(eden.graph._revert_edge_to_vertex_transform,graphs)

        for g in graphs:
            g.graph['layer']=0
        for i in range(self.depth):
            graphs = self.transformers[i].transform(graphs)

        if remove_intermediary_layers:
            graphs= self.do_remove_intermediary_layers(graphs)
        #if self.num_classes == 2:
        #    return graphs,g2
        #else:
        #    return graphs
        
        return graphs

    def  do_remove_intermediary_layers(self, graphs): # transform and remove intermediary layers
        return map(self.remove_intermediary_layers,graphs)

    def remove_intermediary_layers(self,graph):
        def rabbithole(g, n):
            # wenn base graph dann isses halt n
            if 'original' not in g.graph:
                return set([n])

            nodes= g.node[n]['contracted']
            ret=set()
            for no in nodes:
                ret=ret.union(rabbithole(g.graph['original'],no))
            return ret

        for n,d in graph.nodes(data=True):
            d['contracted']= rabbithole(graph,n)
        # ok get rid of intermediary things
        supergraph=graph
        while 'original' in graph.graph:
            graph = graph.graph['original']
        supergraph.graph['original']=graph
        return supergraph


    def re_transform_single(self, graph):
        # the thing has probably expanded edges...
        #print "cascade retransform single was called with this graph:"
        #print utils.ascii.nx_to_ascii(graph,
        #        ymax=30,
        #        edgesymbol='*',
        #        debug="/dev/shm/dump") 

        # thing = eden.graph._revert_edge_to_vertex_transform(graph)  # this line seems pointless since it is repeated in transform
        return self.transform([graph])[0]

def add_fake_abstract_layer(graph):
    '''simply copies the input graph and "associates" the nodes'''
    graph.graph['layer']=0
    g2 = nx.convert_node_labels_to_integers(graph, first_label=max(graph.nodes())+1)
    for a,b in zip(sorted(graph),sorted(g2)):
        g2.node[b]['contracted']=set([a])
        g2.node[b]['importance']=[1.0]
        graph.node[a]['importance']=[1.0]
    g2.graph['original']=graph
    return g2


from graphlearn01.minor.rna.fold import EdenNNF
from eden.graph import _edge_to_vertex_transform
import eden_rna
import graphlearn01.minor.rna as rna

class RNACascade(Cascade):

    def fit(self, eden_sequences):
        self.NNmodel = EdenNNF(n_neighbors=4)
        self.NNmodel.fit(eden_sequences)


        sequences= [b for a,b in eden_sequences]
        seslist = self.NNmodel.transform(sequences)

        return super(self.__class__, self).fit_transform( map(ses_to_graph,seslist) )



    def re_transform_single(self, thing):
        return self.transform([thing])[0]


    def transform(self, eden_sequences_or_graphs,isgraph=True ):


        def refold(graphs):
            if isgraph:
                sequences = map(rna.get_sequence,graphs)
            else:
                sequences= [b for a,b in graphs]
            seslist=self.NNmodel.transform(sequences)
            return map(ses_to_graph,seslist)

        def postprocess_transformer_out(graph):
            # 1. the original graph needs to be directed
            ograph = _edge_to_vertex_transform(graph.graph['original'])
            graph.graph['original'] = rna.expanded_rna_graph_to_digraph(ograph)
            # 2. our convention is, that node ids are not overlapping ,, complying by optimistic renaming..
            graph = nx.convert_node_labels_to_integers(graph,first_label=1000)
            return graph

        undirgraphs =  super(self.__class__, self).transform(refold(eden_sequences_or_graphs))

        return map(postprocess_transformer_out, undirgraphs)


    def fit_transform(self, sequences):
        sequences=list(sequences)
        self.fit(sequences)
        return self.transform(sequences,isgraph=False)



def ses_to_graph(ses):
    structure, energy, sequence = ses
    base_graph = eden_rna.sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=structure)
    #base_graph = _edge_to_vertex_transform(base_graph)  # this keeps the dict
    #base_graph = rna.expanded_rna_graph_to_digraph(base_graph) # i hope this keeps the dict
    base_graph.graph['energy'] = energy
    base_graph.graph['sequence'] = sequence
    base_graph.graph['structure'] = structure
    return base_graph



# PLAN B
'''
from eden.sequence import Vectorizer
from eden.graph import _edge_to_vertex_transform


class RNACascade(Cascade):

    def fit(self, inputs, vectorizer=Vectorizer()):
        # mmmm earlier graphlearn was exprected to pass its vectorizer..
        """
        Parameters
        ----------
        inputs: sequence list
        vectorizer: a vectorizer

        Returns
        -------
        self
        """

        self.vectorizer = vectorizer
        self.NNmodel = EdenNNF(n_neighbors=4)
        self.NNmodel.fit(inputs)
        inputs = [b for a, b in inputs]
        super(self.__class__,self).fit( self.transform(inputs,fold_only=True) )
        return self

    def fit_transform(self, inputs):
        """

        Parameters
        ----------
        inputs: sequences

        Returns
        -------
        many graphdecomposers
        """

        inputs = list(inputs)
        self.fit(inputs)
        inputs = [b for a, b in inputs]
        return self.transform(inputs)

    def re_transform_single(self, graph):
        """

        Parameters
        ----------
        graph: digraph

        Returns
        -------
        graph decomposer
        """

        try:
            sequence = get_sequence(graph)
        except:
            print "forgitransform re_transform_single problem"
            # draw.graphlearn(graph, size=20)
            return None

        #sequence = sequence.replace("F", '')
        trans = self.transform([sequence])[0]
        # if trans._base_graph.graph['energy'] > -10:
        #    return None
        return trans

    def abstract_graph(self, base_graph):

        # create the abstract graph and populate the contracted set
        abstract_graph = rna_forgi.get_abstr_graph(base_graph.graph['structure'], max(base_graph.nodes()) + 1)# DOES NOT EXIST ANYMORE ignore_inserts=self.ignore_inserts)
        abstract_graph = _edge_to_vertex_transform(abstract_graph)
        completed_abstract_graph = rna_forgi.edge_parent_finder(abstract_graph, base_graph)

        # eden is forcing us to set a label and a contracted attribute.. lets do this
        for n, d in completed_abstract_graph.nodes(data=True):
            if 'edge' in d:
                d['label'] = 'e'
        # in the abstract graph , all the edge nodes need to have a contracted attribute.
        # originaly this happens naturally but since we make multiloops into one loop
        # there are some left out
        for n, d in completed_abstract_graph.nodes(data=True):
            if 'contracted' not in d:
                d['contracted'] = set()


        completed_abstract_graph.graph['original']=base_graph
        return completed_abstract_graph


    def transform(self, sequences, fold_only=False ):
        """

        Parameters
        ----------
        sequences : iterable over rna sequences

        Returns
        -------
        list of RnaGraphWrappers
        """
        result = []
        for sequence in sequences:

            # if we eat a tupple, it musst be a (name, sequence) type :)  we only want a sequence
            if type(sequence) == type(()):
                logger.warning('YOUR INPUT IS A TUPPLE, GIVE ME A SEQUENCE, SINCERELY -- YOUR RNA PREPROCESSOR')

            # get structure
            structure, energy, sequence = self.NNmodel.transform_single(('fake', sequence))
            if structure == None:
                result.append(None)
                continue


            # built base_graph
            base_graph = eden_rna.sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=structure)
            base_graph = _edge_to_vertex_transform(base_graph)
            base_graph = expanded_rna_graph_to_digraph(base_graph)
            base_graph.graph['energy'] = energy
            base_graph.graph['sequence'] = sequence
            base_graph.graph['structure'] = structure
            if self.fold_only:
                result.append(base_graph)
            else:
                result.append(self.abstract_graph(base_graph))


        return result
        '''
