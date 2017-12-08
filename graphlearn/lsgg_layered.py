

import lsgg
import lsgg_cip
import transform
import networkx as nx
import copy

import logging
logger = logging.getLogger(__name__)

class lsgg_layered(lsgg.lsgg):

    def _core_substitution(self,graph,cip,cip_):
        return lsgg_cip.core_substitution(graph.graph['original'], cip, cip_)

    def _cip_extraction_given_root(self, graph, root):
        """helper of _cip_extraction. See fit"""
        hash_bitmask = self.decomposition_args['hash_bitmask']
        for radius in self.decomposition_args['radius_list']:
            radius = radius * 2
            for thickness in self.decomposition_args['thickness_list']:
                thickness = thickness * 2
                for e in self.extract_core_and_interface(root_node=root,
                                                 graph=graph,
                                                 radius=radius,
                                                 thickness=thickness,
                                                 hash_bitmask=hash_bitmask):
                    yield e

    def extract_core_and_interface(self,root_node=None,graph=None,radius=None,thickness=None,hash_bitmask=None):

        # get CIP
        basecip = lsgg_cip.extract_core_and_interface(root_node=root_node,
                                                 graph=graph,
                                                 radius=radius,
                                                 thickness=thickness,
                                                 hash_bitmask=hash_bitmask)



        # expand base graph
        orig_graph = graph.graph['original']
        expanded_orig_graph = lsgg_cip._edge_to_vertex(orig_graph)
        lsgg_cip._add_hlabel(expanded_orig_graph)

        # make a copy, collapse core
        expanded_orig_graph_collapsed =  expanded_orig_graph.copy()
        nodes_in_core = list ( reduce(lambda x,y: x|y, [ basecip.graph.node[i]['contracted']
                                for i in basecip.core_nodes if 'edge' not in basecip.graph.node[i] ] ))
        edges_in_core = [n for n,d in expanded_orig_graph_collapsed.nodes(data=True)
                             if 'edge' in d and all([z in nodes_in_core for z in expanded_orig_graph_collapsed.neighbors(n) ])]

        for n in nodes_in_core[1:]+edges_in_core:
            transform.merge_edge(expanded_orig_graph_collapsed,nodes_in_core[0],n)


        # distances...
        dist = nx.single_source_shortest_path_length(expanded_orig_graph_collapsed,
                                        nodes_in_core[0],max(self.decomposition_args['base_thickness_list']))

        # set distance dependant label
        ddl = 'distance_dependent_label'
        for id, dst in dist.items():
            if dst>0:
                expanded_orig_graph.node[id][ddl] = expanded_orig_graph.node[id]['hlabel'] + dst


        for base_thickness in self.decomposition_args['base_thickness_list']:

            interface_nodes = [id for id, dst in dist.items()
                       if 0 < dst <= base_thickness]
            interface_hash = lsgg_cip.graph_hash(expanded_orig_graph_collapsed.subgraph(interface_nodes),
                                hash_bitmask)
            cip=copy.deepcopy(basecip)
            cip.interface_nodes=interface_nodes
            cip.interface_graph = expanded_orig_graph.subgraph(interface_nodes).copy()
            cip.core_nodes=nodes_in_core+edges_in_core
            cip.interface_hash =  lsgg_cip.fast_hash([interface_hash,cip.interface_hash],hash_bitmask)
            cip.graph= expanded_orig_graph.subgraph(interface_nodes+nodes_in_core+edges_in_core).copy()


            #print cip.interface_hash, cip.core_hash, root_node
            yield cip




def test_lsgg_layered():
    import graphlearn as gl
    from transform import cycler
    decomposition_args={ "base_thickness_list":[2],
                        "radius_list": [0],
                        "thickness_list": [1],
                        'hash_bitmask': lsgg._hash_bitmask_}

    lsggg = lsgg_layered(decomposition_args=decomposition_args)
    g = gl.test_get_circular_graph()
    gplus=g.copy()
    gplus.node[0]['label']='weird'
    c=cycler.Cycler()
    g=c.encode_single(g)
    gplus = c.encode_single(gplus)
    assert lsggg.fit([g, gplus, g,gplus])
    assert len(lsggg.neighbors(gplus).next()) == 8







