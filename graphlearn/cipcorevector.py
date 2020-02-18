

from graphlear.lsgg_core_interface_pair import * 
from ego import real_vectorize as rv

# todo https://github.com/fabriziocosta/EGO/blob/master/ego/real_vectorize.py 30:33



class CoreVectorCIP(CoreInterfacePair):

    def __init__(self,core,graph,thickness, node_vectors):
               
        graph, dist =  self.prepare_init(core,graph, thickness)
        self.core_hash = graph_hash(core)
        self.core_nodes = list(core.nodes())
        self.interface = graph.subgraph([id for id, dst in dist.items() if 0 < dst <= thickness])
        get_node_label = lambda id, node: node['hlabel'] + dist[id]
        self.interface_hash = graph_hash(self.interface, get_node_label=get_node_label)
        self.graph = self._get_cip_graph(self.interface, core, graph, dist)

        self.core_vector = self.make_core_vector(core, graph, node_vectors) 


    def make_core_vector(self, core, graph, node_vectors): 
        c_set = set(core.nodes())
        core_ids = [i for i,n in enumerate(graph.nodes()) if n in c_set] 
        return node_vectors[core_ids,:].sum(axis=0)





def vertex_vec(graph, decomposer, bitmask = 2**10-1): 
    '''
        this will generate vectors for all nodes. 
        call this for the whole graph before making a cip
    '''
    encoding, node_ids = rv.node_encode_graph(graph, rv.tz.compose(
        rv.get_subgraphs_from_graph_component, decomposer, rv.convert),
        bitmask=bitmask)

    data_matrix = rv.to_sparse_matrix(encoding, node_ids, bitmask+2)

    return data_matrix



