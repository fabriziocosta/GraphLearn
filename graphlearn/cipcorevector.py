

from graphlearn.lsgg_core_interface_pair import * 
import graphlearn.lsgg_core_interface_pair as cip 
from ego import real_vectorize as rv
from graphlearn import LSGG


# changes tothe base grammar: 
# __init__: 
#     core_vec_decomposer 
#     cipselector 
# neighbors:
#     selectordata


    

class LsggCoreVec(LSGG):

    '''  attaches a vector for each CIP, representing the nodes contained,,, 
        i do this by attaching the vector to the cores when they are generated 
        (so i only need to compute the node vectorisation once)
        and transfer them to the CIPs when make_cip is called
    '''

    def __init__(self,core_vec_decomposer=None,cipselector=[lambda x:x], **kwargs):
        self.core_vec_decomposer = core_vec_decomposer
        self.cipselector = cipselector

        super(LsggCoreVec,self).__init__(**kwargs)


    ##########
    #  step one: cips get vectors 
    #########

    def make_core_vector(self, core, graph, node_vectors): 
        c_set = set(core.nodes())
        core_ids = [i for i,n in enumerate(graph.nodes()) if n in c_set] 
        return node_vectors[core_ids,:].sum(axis=0)

    def _get_cips(self, graph, filter = lambda x:x):
        exgraph = cip._edge_to_vertex(graph)
        matrix = vertex_vec(exgraph, self.core_vec_decomposer) 
        for core in self._get_cores(graph):
            x = self._get_cip(core=core, graph=graph)
            if x and filter(x.graph):
                x.core_vec  = self.make_core_vector(x.graph, exgraph, matrix)
                yield x
    
    def neighbors(self, graph, selectordata, filter = lambda x:True):
        """iterator over all neighbors of graph (that are conceiveable by the grammar)"""
        current_cips = self._get_cips(graph,filter)
        current_cips_congrus = [(current_cip,concip) for current_cip in current_cips 
                for concip in self._get_congruent_cips(current_cip)   ]
        filtered_current_other = self.cipselector(current_cips_congrus,*selectordata)
        for current_cip, congru in filtered_current_other:
            graph_ = self._substitute_core(graph, current_cip, congru)
            if graph_ is not None:
                yield graph_


def vertex_vec(graph, decomposer, bitmask = 2**16-1): 
    '''
        this will generate vectors for all nodes. 
        call this for the whole graph before making a cip
    '''
    encoding, node_ids = rv.node_encode_graph(graph, rv.tz.compose(
        rv.get_subgraphs_from_graph_component, decomposer, rv.convert),
        bitmask=bitmask)

    data_matrix = rv._to_sparse_matrix(encoding, node_ids, bitmask+2)

    return data_matrix


