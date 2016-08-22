'''
automatic minor graph generation
'''
import transform
import decompose
import graphlearn.utils.draw as draw
from sklearn.cluster import MiniBatchKMeans
import sklearn.cluster as cluster


class Cascade():
    def __init__(self,  depth=2,
                        debug=False,
                        multiprocess=True):

        self.depth = depth
        self.debug = debug
        self.multiprocess = multiprocess



    def setup_transformers(self):
        self.transformers = []
        for i in range(self.depth):
            transformer = transform.GraphMinorTransformer(
                group_score_threshold=-.5,
                group_max_size=6,
                group_min_size=3,
                multiprocess=self.multiprocess,
                # cluster_max_members=-1,
                layer=i,
                debug=self.debug)
            self.transformers.append(transformer)

    def fit_transform(self, graphs):
        # INIT
        self.setup_transformers()
        for g in graphs:
            g.graph['layer']=0
        # fitting
        for i in range(self.depth):
            if self.debug:
                print 'graphs at level %d' % i
                draw.graphlearn_layered(graphs[:4])
                #draw.graphlearn_dict(self.transformers[i]..graphclusters, title_key='hash_title',size=2, edge_label='label')

            graphs = self.transformers[i].fit_transform(graphs)
        return graphs

    def fit(self, graphs):
        self.fit_transform(graphs)
        return self

    def transform(self, graphs):
        for g in graphs:
            g.graph['layer']=0
        for i in range(self.depth):
            graphs = self.transformers[i].transform(graphs)
        return graphs

    def transform_ril(self, graphs): # transform and remove intermediary layers
        graphs=self.transform(graphs)
        return map(self.remove_intermediary_layers , graphs )

    def remove_intermediary_layers(self,graph):
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
