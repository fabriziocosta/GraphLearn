'''
automatic minor graph generation
'''
import  transform
import decompose
import graphlearn.utils.draw as draw
from sklearn.cluster import MiniBatchKMeans
import sklearn.cluster as cluster

class Cascade():

    def __init__(self, depth=2, decomposer=decompose.MinorDecomposer(), debug=False):
        self.depth=depth
        self.decomposer=decomposer
        self.debug=debug


    def setup_transformers(self):
        self.transformers = []
        for i in range(self.depth):
            transformer = transform.GraphMinorTransformer(
                #subgraph_cluster= cluster.DBSCAN(),# MiniBatchKMeans(n_clusters=8),
                #save_graphclusters=True,
                group_score_threshold=.1,
                group_max_size=6,
                group_min_size=2,
                #cluster_max_members=-1,
                layer=i,
                debug=self.debug)
            self.transformers.append(transformer)

    def fit_transform(self,graphs):
        # create some transformers
        self.setup_transformers()


        minor_to_nested = lambda x: self.decomposer.make_new_decomposer(x).pre_vectorizer_graph(nested=True)
        graphs=self.write_layer_to_graphs(graphs,0)
        # fitting
        for i in range(self.depth):
            print 'training transformer # %d' % i
            graphs = [minor_to_nested(minor)
                           for minor in self.transformers[i].fit_transform(graphs)]

            if self.debug:
                print 'graphs at this level'
                draw.graphlearn_layered(graphs[:4], vertex_label='layer')
                print 'all the clusters'
                draw.graphlearn_dict(self.transformers[i].graphclusters)
        return graphs



    def fit(self,graphs):
        self.fit_transform(graphs)
        return self

    def transform(self,graphs):
        for i in range(self.depth):
            graphs=[self.decomposer.make_new_decomposer(x).pre_vectorizer_graph(nested=True)
                           for x in self.transformers[i].transform(graphs)]
        return graphs

    def write_layer_to_graphs(self,graphs,layerid):
        # just write the layer id in every node that does not yet have a layer attribute :)
        for graph in graphs:
            for n,d in graph.nodes(data=True):
                    d['layer'] = layerid
            yield graph

