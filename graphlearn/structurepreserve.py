
# VARIANT: 
#  core structure stays the same

from graphlearn.local_substitution_graph_grammar import LocalSubstitutionGraphGrammar as lsgg
import graphlearn.lsgg_core_interface_pair as cip

class StructurePreservingCIP(cip.CoreInterfacePair): 
  
    def __init__(self,core,graph,thickness, preserve_ids=False):
        '''core structure does not change..'''


        super(StructurePreservingCIP,self).__init__(core,graph,thickness)

        structhash = cip.graph_hash(core, get_node_label= lambda i,n: i if preserve_ids else 0)
        self.interface_hash= hash((self.interface_hash,structhash)) 


        '''
        this would also hash node-ids into the interface, but i think this is implied in the structhash already
        def interface_hash(self,interface):
            get_node_label = lambda id, node: id if self.preserve_ids else node['ilabel'] + dist[id]
            interface_hash = cip.graph_hash(self.interface, get_node_label=get_node_label)
            return interface_hash
        '''



class StructurePreservingGrammar(lsgg):
    def __init__(self,preserve_ids, **kwargs):
        super(StructurePreservingGrammar,self).__init__(**kwargs)
        self.preserve_ids= preserve_ids

    def _make_cip(self, core=None, graph=None): 
        return StructurePreservingCIP(core=core, graph=graph, thickness=self.thickness, 
                preserve_ids=self.preserve_ids)
