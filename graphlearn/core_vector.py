

from graphlearn.lsgg_core_interface_pair import * 
from ego import real_vectorize as rv
from graphlearn import LSGG


# Todo, make_core_vector generates a matrix, maybe a vector is more usefull


class LsggCoreVec(LSGG):

    '''  attaches a vector for each CIP, representing the nodes contained,,, 
        i do this by attaching the vector to the cores when they are generated 
        (so i only need to compute the node vectorisation once)
        and transfer them to the CIPs when make_cip is called
    '''

    def __init__(self,decomposer=None, **kwargs):
        self.decomposer = decomposer
        super(LsggCoreVec,self).__init__(**kwargs)


    ##########
    #  step one: cores get vectors attached
    #########

    def _get_cores(self, graph):
        cores= super(LsggCoreVec,self)._get_cores(graph)
        self.attach_vectors(cores, graph)
        return cores


    def attach_vectors(self, cores, graph):
        matrix = vertex_vec(graph, self.decomposer) # should put decomposer in init...
        for core in cores:
            core.core_vec = self.make_core_vector(core, graph, matrix)

    def make_core_vector(self, core, graph, node_vectors): 
        c_set = set(core.nodes())
        core_ids = [i for i,n in enumerate(graph.nodes()) if n in c_set] 
        return node_vectors[core_ids,:].sum(axis=0)


    ###
    # step 2: vector gets transfered to cip
    ######
    
    def _get_cip(self, core=None, graph=None):
        cip = super(LsggCoreVec, self)._get_cip(core, graph)
        if cip:
            cip.core_vec= core.core_vec 
            return cip
    #####
    # step 3: one should probably overwrite get_congruent cips to actually use the new vectors
    # TODO
    # neighbors takes 2 more args 
    # these are passed to new unctions that i got in init
    # one should filter cores in get_cips (depending on the prev mentioned arg)
    # one should filter congruent cips (the other arg)
    ######

    def _get_congruent_cips(self, cip):
        """all cips in the grammar that are congruent to cip in random order.
        congruent means they have the same interface-hash-value"""
        cips = self.productions.get(cip.interface_hash, {}).values()
        cips_ = [cip_ for cip_ in cips if cip_.core_hash != cip.core_hash]

        # quick and dirty hack because i am too lazy to implement step 3:
        self.filtercips(cips_,cip) 
        return cips_

    

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


