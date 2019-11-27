




from graphlearn import lsgg 
from graphlearn import lsgg_cip
import networkx as nx

# lsgg_life -> Loose InterFacE around the actual interface


class LIFE(lsgg.lsgg):

    def _extract_core_and_interface(self, **kwargs):
        return extract_core_and_interface(thickness_life=self.decomposition_args['thickness_life'] , **kwargs)

    
    def _congruent_cips(self, cip):
        cips = self.productions.get(cip.interface_hash, {}).values()
        cips_ = [(cip_,cip_.life_vector*cip.life_vector) for cip_ in cips if cip_.core_hash != cip.core_hash]

	
        print(cips_)
        random.shuffle(cips_)
        return cips_ 




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
                                         thickness+radius)}

    core_nodes = [id for id, dst in dist.items() if dst <= radius]
    interface_nodes = [id for id, dst in dist.items()
                       if radius < dst <= radius + thickness]

    normal_cip =  _finalize_cip(root_node,graph,radius,thickness,dist,core_nodes,interface_nodes)


    # NOW COMES THE LIFE PART

    life_nodes = [id for id, dst in dist.items()
                       if (radius+thickness) < dst <= (radius + thickness+ thickness_life)]

    life_graph = graph.subgraph(life_nodes) 
    normal_cip.life_vector = lsgg_cip.eg.vectorize(nx.Graph(life_graph))

    return normal_cip

