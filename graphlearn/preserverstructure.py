
# VARIANT: 
#  core structure stays the same

class StructurePreservingCIP(CoreInterfacePair): 
  
    def __init__(self,core,graph,thickness, preserve_ids=False):
            '''core structure does not change '''
            
            # do some of the basic init:
            graph, dist =  self.prepare_init(core,graph, thickness)
            self.core_hash = graph_hash(core)
            self.core_nodes = list(core.nodes())
            self.interface = graph.subgraph([id for id, dst in dist.items() if 0 < dst <= thickness])


            # interface and core hashes need  special attention:
            
            # interface_hash might want to preserve the ids
            get_node_label = lambda id, node: id if preserve_ids else node['hlabel'] + dist[id]
            self.interface_hash = graph_hash(self.interface, get_node_label=get_node_label)
            self.graph = self._get_cip_graph(self.interface, core, graph, dist)

            # in the core we calculate a structure hash that is appended to the interfacehash
            structhash = graph_hash(core, get_node_label= lambda i,n: i if preserve_ids else 0)
            self.interface_hash= hash(self.interface_hash,structhash) 


