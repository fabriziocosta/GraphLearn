

from graphlear.lsgg_core_interface_pair import * 
from ego import real_vectorize as rv


class lsgg_extension:

    def _get_cores(self, graph):
        cores = [ core for core in lsgg_core_interface_pair.get_cores(graph, self.radii) if core]
        self.attach_vectors(cores, graph)
        return cores


    def _make_cip(self, core=None, graph=None):
        cip = super(lsgg_extension, self)._make_cip(core, graph) 
        if cip:
            cip.core_vec= core.core_vec 
            return cip
        

    def attach_vectors(self, cores, graph):
        matrix = vertex_vec(graph, self.decomposer) # should put decomposer in init...
        for core in cores:
            core.core_vec = self.make_core_vector(core, graph, node_vectors)


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



