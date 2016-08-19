'''
automatic minor graph generation
'''
import transform
import decompose
import graphlearn.utils.draw as draw
from sklearn.cluster import MiniBatchKMeans
import sklearn.cluster as cluster


class Cascade():
    def __init__(self, depth=2, decomposer=decompose.MinorDecomposer(), debug=False, multiprocess=False):
        self.depth = depth
        self.decomposer = decomposer
        self.debug = debug
        self.multiprocess = multiprocess

    def setup_transformers(self):
        self.transformers = []
        for i in range(self.depth):
            transformer = transform.GraphMinorTransformer(
                # subgraph_cluster= cluster.DBSCAN(),# MiniBatchKMeans(n_clusters=8),
                # save_graphclusters=True,
                group_score_threshold=-.5,
                group_max_size=6,
                group_min_size=3,
                multiprocess=self.multiprocess,
                # cluster_max_members=-1,
                layer=i,
                debug=self.debug)
            self.transformers.append(transformer)

    def fit_transform(self, graphs):
        # create some transformers
        self.setup_transformers()
        graphs = self.write_layer_to_graphs(graphs, 0)
        # fitting
        for i in range(self.depth):
            print 'training transformer # %d' % i
            # graphs = [minor_to_nested(minor)
            graphs = [minor
                      for minor in self.transformers[i].fit_transform(graphs)]

            if self.debug:
                print 'graphs at this level'
                # draw.graphlearn_layered(graphs[:4], vertex_label='layer')
                print 'all the clusters'
                draw.graphlearn_dict(self.transformers[i].graphclusters, title_key='hash_title',size=2, edge_label='label')
        return graphs

    def fit(self, graphs):
        self.fit_transform(graphs)
        return self

    def transform(self, graphs):
        graphs = list(self.write_layer_to_graphs(graphs, 0))
        for i in range(self.depth):
            graphs = self.transformers[i].transform(graphs)
        return graphs

    def transform_ril(self, graphs): # transform and remove intermediary layers
        graphs=self.transform(graphs)
        return map(self.remove_intermediary_layers , graphs )


    def remove_intermediary_layers(self,graph):
        # write the contracted set to the master layer

        def rabbithole(g, n):
            # wenn base graph dann isses halt n
            if 'original' not in g.graph:
                return [n]

            nodes= g.node[n]['contracted']
            ret=[]
            for no in nodes:
                ret+=rabbithole(g.graph['original'],no)
            return ret

        for n,d in graph.nodes(data=True):
            d['contracted']= rabbithole(graph,n)

        # ok get rid of intermediary things
        supergraph=graph
        while 'original' in graph.graph:
            graph = graph.graph['original']
        supergraph.graph['original']=graph
        return supergraph



    def write_layer_to_graphs(self, graphs, layerid):
        # just write the layer id in every node that does not yet have a layer attribute :)
        for graph in graphs:
            for n, d in graph.nodes(data=True):
                d['layer'] = layerid
            yield graph
