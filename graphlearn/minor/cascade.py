'''
automatic minor graph generation
'''
import  transform
import decompose
import graphlearn.utils.draw as draw
from sklearn.cluster import MiniBatchKMeans
import sklearn.cluster as cluster

class cascade():

    def __init__(self, depth=2, decomposer=decompose.MinorDecomposer()):
        self.depth=depth
        self.decomposer=decomposer


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
                debug=False)

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
            #draw.graphlearn_layered(graphs[:4], vertex_label='layer')

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

'''
TODO add layer annotation, also do grpahs know their minor master?
def fix_graph(g):
    for a,b,d in g.edges(data=True):
        if 'label' not in d:
            d['label']=''
    for a,d in g.nodes(data=True):
        if 'contracted' not in d:
            d['layer']='0'
        else:
            d['layer']='1'
    return g

'''
class mydecomposer(decompose.MinorDecomposer):
    # we need to mod the decomposer a littlebit:

    # first problem: layer info eintragen. vlt nicht hier?
    # zweites problem:
    pass