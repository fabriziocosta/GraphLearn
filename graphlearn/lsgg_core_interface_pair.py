import eden.graph as eg
import structout as so 
from networkx.algorithms import isomorphism as iso
import networkx as nx
import logging

# nx 2.2 has this:
from networkx.algorithms.shortest_paths.unweighted import _single_shortest_path_length as short_paths
logger = logging.getLogger(__name__)



def _add_hlabel(graph):
    eg._label_preprocessing(graph)

def _edge_to_vertex(graph):
    return eg._edge_to_vertex_transform(graph)


def graph_hash(graph, get_node_label=lambda id, node: node['hlabel']):
    """
    calculate a hash of a graph
    """
    node_neighborhood_hashes = {n: _graph_hash_neighborhood(graph, n, get_node_label) for n in graph.nodes()}

    edge_hash = lambda a, b: hash((min(a, b), max(a, b)))
    l = [edge_hash(node_neighborhood_hashes[a],
                   node_neighborhood_hashes[b]) for (a, b) in graph.edges()]
    l.sort()

    isolates = [n for (n, d) in graph.degree if d == 0]
    z = [get_node_label(node_id, graph.nodes[node_id]) for node_id in isolates]
    z.sort()
    return hash(tuple(l + z))


def _graph_hash_neighborhood(graph, node, get_node_label=lambda id, node: node['hlabel']):
    d = nx.single_source_shortest_path_length(graph, node, 5)
    l = [hash((get_node_label(nid, graph.nodes[nid]), dis)) for nid, dis in d.items()]
    l.sort()
    return hash(tuple(l))






################
#  decompose
###############

class CoreInterfacePair:
    """
    this is referred to throughout the code as cip
    it contains the cip-graph and several pieces of information about it.
    """


    def __init__(self,core,graph,thickness):
            '''
            graph = _edge_to_vertex(graph)
            _add_hlabel(graph)
            _add_hlabel(core)
            dist = {a: b for (a, b) in short_paths(graph, core.nodes(), thickness)}
            self.count=0
            '''
            
            # preprocess, distances of core neighborhood, init counter
            graph, dist =  self.prepare_init(core,graph, thickness)

            # core
            self.core_hash = graph_hash(core)
            self.core_nodes = list(core.nodes())

            # interface
            self.interface = graph.subgraph([id for id, dst in dist.items() if 0 < dst <= thickness])
            get_node_label = lambda id, node: node['hlabel'] + dist[id]
            self.interface_hash = graph_hash(self.interface, get_node_label=get_node_label)

            # cip
            self.graph = self._get_cip_graph(self.interface, core, graph, dist)

    def prepare_init(self, core, graph, thickness): 
        # preprocess, distances of core neighborhood, init counter
        graph = _edge_to_vertex(graph)
        _add_hlabel(graph)
        _add_hlabel(core)
        dist = {a: b for (a, b) in short_paths(graph, core.nodes(), thickness)}
        self.count=0
        return graph, dist 

    def _get_cip_graph(self,interface, core, graph, dist):
        cip_graph = graph.subgraph( list(core.nodes()) + list(interface.nodes()))
        ddl = 'distance_dependent_label'
        for no in interface.nodes():
            cip_graph.nodes[no][ddl] = cip_graph.nodes[no]['hlabel'] + dist[no]
        return cip_graph


    def ascii(self): 
        '''return colored cip'''
        return so.graph.make_picture(self.graph, color=[ self.core_nodes , list(self.interface.nodes())  ])

    def __str__(self):
        return 'cip: int:%d, cor:%d, rad:%d, size:%d' % \
               (self.interface_hash, 
                       self.core_hash, 
                       self.radius, 
                       len(self.core_nodes))



# VARIANT: 
# 1. core structure stays the same (bonus: keep node-ids) 
# 2. cores have vector attached to predict the impact on the vectorized graph 


class StructurePreservingCIP(CoreInterfacePair): 
    def __init__(self,core,graph,thickness, preserveid = False):
        super(StructurePreservingCIP,self).__init__(core,graph, thickness)
        # preserve structure:
        getlabel =  lambda id, node: id if preserveid else '1337'
        self.interface_hash = hash(self.interface_hash,
                graph_hash(core, get_node_label=getlabel) )







#########
# CORES
#########
def get_cores(graph, radii):
    exgraph = _edge_to_vertex(graph)
    edgeout =  max(radii) % 2 # is 1 if the outermost node is an edge
    for root in graph.nodes():
        #id_dst = short_paths(exgraph,[root], max(radii)+edgeout)
        id_dst = {a: b for (a, b) in short_paths(exgraph, [root], max(radii)+edgeout)}
        for r in radii:
            if r % 2 == 0:
                yield exgraph.subgraph([id for id,dst in id_dst.items() if dst <= r])
            else:
                # use the
                assert False, "not implemented, outermost node is an edge, we would like to put this in the core... "


'''

# use this to implement the above assert error

def get_cores_mv_to_core(interface_nodes, core_nodes, graph):
    """
     this is relevant when  1. edges are root  OR 2. interface sizes are non standard
     the problem-case is this: an interface-edge-node is connected to cores, so using it for interface
     isomorphismchecks is nonse.
    """
test = lambda idd: 2==sum([ neigh in core_nodes for neigh in graph.neighbors(idd) ])
mv_to_core = {idd:0 for idd in interface_nodes if "edge" in graph.nodes[idd] and test(idd)}
if mv_to_core:
    core_nodes+= list(mv_to_core.keys())
    interface_nodes = [i for i in interface_nodes if i not in mv_to_core ]
return core_nodes, interface_nodes 
'''


######
# compose
######

def find_all_isomorphisms(interface_graph, congruent_interface_graph):
    ddl = 'distance_dependent_label'
    label_matcher = lambda x, y: x[ddl] == y[ddl]  # and \ x.get('shard', 1) == y.get('shard', 1)
    return iso.GraphMatcher(interface_graph, congruent_interface_graph, node_match=label_matcher).match()

def substitute_core(graph, cip, congruent_cip):
    
    # expand edges and remove old core
    graph = _edge_to_vertex(graph)
    graph.remove_nodes_from(cip.core_nodes)

    # relabel the nodes in the congruent cip such that the interface node-ids match with the graph and the
    # core ids dont overlap
    interface_map = next(find_all_isomorphisms(congruent_cip.interface, cip.interface))
    if len(interface_map) != len(cip.interface):
        logger.log(10, "isomorphism failed, likely due to hash collision")
        return None
    maxid = max(graph.nodes())
    core_rename= { c: i+maxid+1 for i,c in enumerate(congruent_cip.core_nodes) }
    interface_map.update(core_rename)
    newcip = nx.relabel_nodes(congruent_cip.graph, interface_map,copy=True)
    

    # compose and undo edge expansion
    graph2= nx.compose(graph,newcip)

    # if the reverserion fails, you use a wrong version of eden, where
    # expansion requires that edges are indexed by (0..n-1)
    return  eg._revert_edge_to_vertex_transform(graph2)
    '''
    except Exception as e:
        print(str(e))
        print('imap:', interface_map)
        print(newcip.nodes()) 
        print(graph.nodes()) 
        so.gprint(graph, size=30, nodelabel=None, color=[[v for v in interface_map.values() if v in graph.nodes()]])
        so.gprint(newcip,nodelabel=None, color=[list(core_rename.values())])
        so.gprint(graph2, size=30, nodelabel=None, color=[list(interface_map.values())])
        so.graph.ginfo(graph2) 
        so.gprint(graph_orig, size=30,nodelabel=None)

    return ret
    '''
