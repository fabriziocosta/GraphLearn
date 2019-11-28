from graphlearn import lsgg 
import structout as so
from graphlearn import lsgg_cip
import networkx as nx
import numpy as np
import random
# lsgg_life -> Loose InterFacE around the actual interface
# mmm lscc LOCO loose context is also ok... 


class LIFE(lsgg.lsgg):

    def _extract_core_and_interface(self, **kwargs):
        return extract_core_and_interface(thickness_life=self.decomposition_args['thickness_life'] , **kwargs)
    
    def _congruent_cips(self, cip):
        cips = self.productions.get(cip.interface_hash, {}).values()
        def dist(a,b):
            if type(a)==np.ndarray and type(b)==np.ndarray:
                return np.dot(a,b.T)[0][0]
            else: 
                return 0
        cips_ = [(cip_,dist(cip_.life_vector,cip.life_vector)) for cip_ in cips if cip_.core_hash != cip.core_hash]

        random.shuffle(cips_)
        return [ c for c,i in  cips_ ]

def extract_core_and_interface(root_node=None,
                               graph=None,
                               radius=None,
                               thickness=None,
			       thickness_life=2):
   
    # MAKE A NORMAL CIP AS IN LSGG_CIP
    graph =  lsgg_cip._edge_to_vertex(graph)
    lsgg_cip._add_hlabel(graph)
    dist = {a:b for (a,b) in lsgg_cip.short_paths(graph,
                                         root_node if isinstance(root_node,list) else [root_node],
                                         thickness+radius+thickness_life)}

    core_nodes = [id for id, dst in dist.items() if dst <= radius]
    interface_nodes = [id for id, dst in dist.items()
                       if radius < dst <= radius + thickness]

    normal_cip =  lsgg_cip._finalize_cip(root_node,graph,radius,thickness,dist,core_nodes,interface_nodes)


    # NOW COMES THE LIFE PART

    life_nodes = [id for id, dst in dist.items()
                       if (radius+thickness) < dst <= (radius + thickness+ thickness_life)]

    life_graph = graph.subgraph(life_nodes) 
    
    loosecontext = nx.Graph(life_graph)
    nn = loosecontext.number_of_nodes() > 2
    # eden doesnt like empty graphs, they should just be a 0 vector... 
    normal_cip.life_vector = lsgg_cip.eg.vectorize([loosecontext])[0].toarray() if nn else None

    return normal_cip

