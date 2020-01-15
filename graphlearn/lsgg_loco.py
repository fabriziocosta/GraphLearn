from graphlearn import lsgg 
import structout as so
from graphlearn import lsgg_cip
import networkx as nx
import numpy as np
import random
import  scipy.sparse as sparse
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
            return a.dot(b.T)[0,0]
            
        #cips_ = [(cip_,max([dist(cip.loco_vectors[0],b) for b in cip_.loco_vectors]))
        cips_ = [(cip_,np.max( cips.loco_vectors.dot(cip.loco_vectors)))
                     for cip_ in cips if cip_.core_hash != cip.core_hash]
         
        #ret = [ c for c,i in  cips_ if i > self.decomposition_args['loco_minsimilarity'] ]
        #if len(ret)<1 : logger.info( [b for a,b in cips_]  )

        for cip_, di in cips_:
            if di > 0.0:
                cip_.locosimilarity=di
                yield cip_
    
    def _neighbors_sample_order_proposals(self,subs): 
        sim = [c[1].locosimilarity for c in subs ]
        logger.log(29, "cips_ priosim scores:")
        logger.log(29, sim)
        p_size = [a*b for a,b in zip (self.get_size_proba(subs),sim)]
        return self.order_proba(subs, p_size)


    def _add_cip(self, cip):
                    
        grammarcip = self.productions[cip.interface_hash].setdefault(cip.core_hash, cip)
        grammarcip.count+=1
        if not grammarcip.locohash.intersection(cip.locohash): 
            grammarcip.loco_vectors= sparse.vstack( (grammarcip.loco_vectors,  cip.loco_vectors))
            grammarcip.locohash= grammarcip.locohash.union(cip.locohash)


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
                                         radius+max(thickness_loco,thickness))}

    core_nodes = [id for id, dst in dist.items() if dst <= radius]
    interface_nodes = [id for id, dst in dist.items()
                       if radius < dst <= radius + thickness]

    normal_cip =  lsgg_cip._finalize_cip(root_node,graph,radius,thickness,dist,core_nodes,interface_nodes)


    # NOW COMES THE loco PART

    loco_nodes = [id for id, dst in dist.items()
                       if radius < dst <= (radius + thickness_loco)]

    loco_graph = graph.subgraph(loco_nodes) 
    
    loosecontext = nx.Graph(loco_graph)
    if loosecontext.number_of_nodes() > 2:
        normal_cip.loco_vectors = lsgg_cip.eg.vectorize([loosecontext])
        normal_cip.loco_hash = set([ lsgg_cip.graph_hash(loosecontext)] ) 
        return normal_cip
    return None



