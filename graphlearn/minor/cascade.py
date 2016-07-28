'''
automatic minor graph generation
'''
import transform
import decompose

from sklearn.cluster import MiniBatchKMeans

class cascade():

    def __init__(self, depth=2, decomposer=decompose.MinorDecomposer()):
        self.depth=depth
        self.decomposer=decomposer


    def setup_transformers(self):
        self.transformers = []
        for i in range(self.depth):
            transformer = transform.GraphMinorTransformer(
                subgraph_name_estimator=MiniBatchKMeans(n_clusters=5),
                save_graphclusters=True,
                group_score_threshold=2,
                group_min_size=2,
                cluster_max_members=-1)
            self.transformers.append(transformer)

    def fit_transform(self,graphs):
        # create some transformers
        self.setup_transformers()

        # fitting
        for i in range(self.depth):
            graphs = [self.decomposer.make_new_decomposer(x).pre_vectorizer_graph(nested=True)
                           for x in self.transformers[i].fit_transform(graphs)]
        return graphs


    def fit(self,graphs):
        self.fit_transform(graphs)
        return self

    def transform(self,graphs):
        for i in range(self.depth):
            graphs=[self.decomposer.make_new_decomposer(x).pre_vectorizer_graph(nested=True)
                           for x in self.transformers[i].transform(graphs)]
        return graphs

'''
TODO add layer annotation, also do grpahs know their minor master
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
