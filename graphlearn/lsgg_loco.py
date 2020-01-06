from graphlearn import lsgg 
import structout as so
from graphlearn import lsgg_cip
import networkx as nx
import numpy as np
import random
from scipy.sparse import csr_matrix
import logging
logger = logging.getLogger(__name__)
from sklearn.metrics.pairwise import cosine_similarity

# lsgg_loco   loose context around the interface

class LOCO(lsgg.lsgg):

    def _extract_core_and_interface(self, **kwargs):
        return extract_core_and_interface(thickness_loco=2*self.decomposition_args['thickness_loco'],
                                          **kwargs)
    
    def _congruent_cips(self, cip):
        cips = self.productions.get(cip.interface_hash, {}).values()
        def dist(a,b):
            if type(a)==csr_matrix and type(b)==csr_matrix:
                return a.dot(b.T)[0,0]
            elif type(a)!=csr_matrix and type(b)!=csr_matrix:
                return 1 # both None
            else: 
                return 0 # one is none
        
        #print ('cip',cip.loco_vectors,)
        #print('congruent:',list(cips)[0].loco_vectors)
        #cips_ = [(cip_,max([dist(cip_.loco_vectors,b) for b in cip.loco_vectors]))
        cips_ = [(cip_,max([dist(cip.loco_vectors[0],b) for b in cip_.loco_vectors]))
                     for cip_ in cips if cip_.core_hash != cip.core_hash]

        
        random.shuffle(cips_)
        ret = [ c for c,i in  cips_ if i > self.decomposition_args['loco_minsimilarity'] ]
        #if len(ret)<1 : logger.info( [b for a,b in cips_]  )
        return ret

    def _add_cip(self, cip):

        def same(a,b):
            if type(a)==csr_matrix and type(b)==csr_matrix:
                return np.array_equal(a.data,b.data)and np.array_equal(a.indptr,b.indptr)and np.array_equal(a.indices,b.indices)
            if type(a)!=csr_matrix and type(b)!=csr_matrix:
                return True
            if type(a)!=csr_matrix or  type(b)!=csr_matrix:
                return False

            print("OMGWTFBBQ")
                    
        grammarcip = self.productions[cip.interface_hash].setdefault(cip.core_hash, cip)
        grammarcip.count+=1
        if not any([same(cip.loco_vectors[0],x) for x in  grammarcip.loco_vectors]):
            grammarcip.loco_vectors+=cip.loco_vectors


def extract_core_and_interface(root_node=None,
                               graph=None,
                               radius=None,
                               thickness=None,
			       thickness_loco=2):
   
    # MAKE A NORMAL CIP AS IN LSGG_CIP
    graph =  lsgg_cip._edge_to_vertex(graph)
    lsgg_cip._add_hlabel(graph)
    dist = {a:b for (a,b) in lsgg_cip.short_paths(graph,
                                         root_node if isinstance(root_node,list) else [root_node],
                                         thickness+radius+thickness_loco)}

    core_nodes = [id for id, dst in dist.items() if dst <= radius]
    interface_nodes = [id for id, dst in dist.items()
                       if radius < dst <= radius + thickness]

    normal_cip =  lsgg_cip._finalize_cip(root_node,graph,radius,thickness,dist,core_nodes,interface_nodes)


    # NOW COMES THE loco PART

    loco_nodes = [id for id, dst in dist.items()
                       if (radius+thickness) < dst <= (radius + thickness+ thickness_loco)]

    loco_graph = graph.subgraph(loco_nodes) 
    
    loosecontext = nx.Graph(loco_graph)
    nn = loosecontext.number_of_nodes() > 2
    # eden doesnt like empty graphs, they should just be a 0 vector... 
    normal_cip.loco_vectors = [lsgg_cip.eg.vectorize([loosecontext])] if nn else [None]

    return normal_cip

