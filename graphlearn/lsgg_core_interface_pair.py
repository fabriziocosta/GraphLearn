import eden.graph as eg
from networkx.algorithms import isomorphism as iso
import networkx as nx
import logging

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

    
    PARAMS:
    core: an 'expanded' subgraph of graph
    graph: an unexpanded graph
    thickness: absolute thickness on expanded Graph


    ATTRIBUTES:
    graph: expanded CIP-graph
    core_hash: hash of the core, used for filtering duplicates
    core_nodes: list of node-ids in the core
    interface: interface graph, augmented with a distance_dependant_label
        this label ensures that the correct isomorphism is found when 
        substituting
    interface_hash: finding congruent cips
    count: when this cip is placed in a grammar, we will count the number of
        occurences
    
    """


    def __init__(self,core,graph,thickness):
                     
            # preprocess, distances of core neighborhood, init counter
            exgraph, dist = self.initialize_params(core,graph, thickness)

            # core and graph, no surprises there
            self.core_hash = graph_hash(core)
            self.core_nodes = list(core.nodes())
            self.graph = exgraph.subgraph([id for id, dst in dist.items() if dst <= thickness])
            # interface and hash are more tricky...
            self.interface,  self.interface_hash  = self.make_interface(exgraph, dist, self.core_nodes,self.graph)


    def make_interface(self, exgraph, dist, core_nodes, cipgraph):
        # generate graph
        interface = exgraph.subgraph([n for n,dst in dist.items() if dst > 0])

        # adjust node-labels for matching and hashing...
        for no in interface.nodes():
            interface.nodes[no]['ilabel'] = interface.nodes[no]['hlabel'] + dist[no]
            if dist[no] == 1 and 'edge' in interface.nodes[no] and \
                    2==sum([cipgraph.has_edge(i,no) for i in core_nodes]):

                interface.nodes[no]['ilabel'] += 1337

        return interface, self.interface_hash(interface)

    def interface_hash(self,interface):
        get_node_label = lambda id, node: node['ilabel']
        interface_hash = graph_hash(interface, get_node_label=get_node_label)
        return interface_hash


    def initialize_params(self, core, graph, thickness):
        # preprocess, distances of core neighborhood, init counter
        exgraph = _edge_to_vertex(graph)
        _add_hlabel(exgraph)
        _add_hlabel(core)
        dist = {a: b for (a, b) in short_paths(exgraph, core.nodes(), thickness)}
        self.count=0
        return exgraph, dist



    def ascii(self): 
        '''return colored cip-graph'''
        import structout as so 
        return so.graph.make_picture(self.graph, color=[ self.core_nodes , list(self.interface.nodes())  ])

    def __str__(self):
        return 'cip: int:%d, cor:%d, size:%d' % \
               (self.interface_hash, 
                       self.core_hash, 
                       len(self.core_nodes))

#########
# CORES
#########
def get_cores(graph, radii):
    exgraph = _edge_to_vertex(graph)
    for root in graph.nodes():
        id_dst = {node: dis for (node, dis) in short_paths(exgraph, [root], max(radii)+1)}
        for r in radii:
            nodeset = get_node_set(id_dst,r, exgraph)
            if len(nodeset) < len(exgraph):
                yield  exgraph.subgraph(nodeset)
            #yield  exgraph.subgraph([id for id,dst in id_dst.items() if dst <= r ])
            #print (root, id_dst)
            #so.gprint(res)

def get_node_set(id_dst, r, graph):
    # a node is in the core when dist <= r or it is an edge and is twice connected to nodes in core
    border = {node for node,dis in id_dst.items() if dis == r} 
    return [id for id,dst in id_dst.items() if (dst <= r or edgetest(border,id, graph))]

def edgetest(border, id,g):
   res=  (2 == sum([ g.has_edge( id,b  ) for b in border]))# and "edge" in graph.nodes[id] 
   return res




######
# compose
######

def find_all_isomorphisms(interface_graph, congruent_interface_graph):
    label_matcher = lambda x, y: x['ilabel'] == y['ilabel']  # and \ x.get('shard', 1) == y.get('shard', 1)
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
    
    maxid = max(graph.nodes()) # if we die here, likely the cip covers the whole graph 
    core_rename= { c: i+maxid+1 for i,c in enumerate(congruent_cip.core_nodes) }
    interface_map.update(core_rename)
    newcip = nx.relabel_nodes(congruent_cip.graph, interface_map,copy=True)
    

    # compose and undo edge expansion
    graph2= nx.compose(graph,newcip)

    # if the reverserion fails, you use a wrong version of eden, where
    # expansion requires that edges are indexed by (0..n-1)
    return   eg._revert_edge_to_vertex_transform(graph2)

    '''
    except Exception as e:
        print(str(e))
        print('imap:', interface_map)
        print(newcip.nodes()) 
        print(graph.nodes()) 

        ZOOM2  = [a for (a, b) in short_paths(graph2, interface_map.values(), 5)]
        print("substituted")
        so.gprint(graph2.subgraph(ZOOM2), size=30, nodelabel=None, color=[list(interface_map.values())])
        color = [v for v in interface_map.values() if v in graph.nodes()]
        ZOOM  = [a for (a, b) in short_paths(graph, color,  5)]
        print("orig, stuff removed")
        so.gprint(graph.subgraph(ZOOM), size=30, nodelabel=None, color=[color])
        print("cip (relabeled)")
        so.gprint(newcip,nodelabel=None, color=[list(core_rename.values())])
        so.graph.ginfo(graph2.subgraph(ZOOM2))

    return ret
    '''
