

from graphlearn.test import transformutil
import copy
import functools
from graphlearn import lsgg
from graphlearn import lsgg_cip
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import logging
logger = logging.getLogger(__name__)



class lsgg_locolayer(lsgg.lsgg):


    # theese should work the same way as loco i assume
    def _add_cip(self, cip):

        def same(a, b):
            if type(a) == csr_matrix and type(b) == csr_matrix:
                return np.array_equal(a.data, b.data) and np.array_equal(a.indptr, b.indptr) and np.array_equal(
                    a.indices, b.indices)
            if type(a) != csr_matrix and type(b) != csr_matrix:
                return True
            if type(a) != csr_matrix or type(b) != csr_matrix:
                return False

            print("OMGWTFBBQ")

        grammarcip = self.productions[cip.interface_hash].setdefault(cip.core_hash, cip)
        grammarcip.count += 1
        if not any([same(cip.loco_vectors[0], x) for x in grammarcip.loco_vectors]):
            grammarcip.loco_vectors += cip.loco_vectors



    def _congruent_cips(self, cip):
        cips = self.productions.get(cip.interface_hash, {}).values()

        def dist(a, b):
            if type(a) == csr_matrix and type(b) == csr_matrix:
                return a.dot(b.T)[0, 0]
            elif type(a) != csr_matrix and type(b) != csr_matrix:
                return 1  # both None
            else:
                return 0  # one is none

        # print ('cip',cip.loco_vectors,)
        # print('congruent:',list(cips)[0].loco_vectors)
        # cips_ = [(cip_,max([dist(cip_.loco_vectors,b) for b in cip.loco_vectors]))
        cips_ = [(cip_, max([dist(cip.loco_vectors[0], b) for b in cip_.loco_vectors]))
                 for cip_ in cips if cip_.core_hash != cip.core_hash]

        # ret = [ c for c,i in  cips_ if i > self.decomposition_args['loco_minsimilarity'] ]
        # if len(ret)<1 : logger.info( [b for a,b in cips_]  )
        sumdists = sum([b for cip_, b in cips_])
        if sumdists == 0.0:
            return
        for cip_, di in cips_:
            if di > 0.0:
                cip_.locosimilarity = di / sumdists
                yield cip_

    def _neighbors_sample_order_proposals(self, subs):
        sim = [c[1].locosimilarity for c in subs]
        suu = sum(sim)
        samples = np.random.choice(list(range(len(subs))), size=len(subs), replace=False,
                                   p=[x / suu for x in sim])
        return [subs[i] for i in samples]

    # this should work like layered
    def _core_substitution(self,graph,cip,cip_):
        return lsgg_cip.core_substitution(graph.graph['original'], cip, cip_)



    def _cip_extraction_given_root(self, graph, root):
        """helper of _cip_extraction. See fit"""
        for radius in self.decomposition_args['radius_list']:
            radius = radius * 2
            for thickness in self.decomposition_args['thickness_list']:
                thickness = thickness * 2
                # note that this loop is different from the parent class :) 
                for e in self._extract_core_and_interface(root_node=root,
                                                 graph=graph,
                                                 radius=radius,
                                                 thickness=thickness):
                    yield e

    def _extract_core_and_interface(self,root_node=None,graph=None,radius=None,thickness=None,hash_bitmask=None):
        basecip = extract_core_and_interface(root_node=root_node,
                                                 graph=graph,
                                                 radius=radius,
                                                 thickness=thickness,
                                                 thickness_loco = self.decomposition_args['thickness_loco'])



        # expand base graph
        orig_graph = graph.graph['original']
        expanded_orig_graph = lsgg_cip._edge_to_vertex(orig_graph)

        lsgg_cip._add_hlabel(expanded_orig_graph)

        # make a copy, collapse core
        expanded_orig_graph_collapsed =  expanded_orig_graph.copy()
        nodes_in_core = list ( functools.reduce(lambda x,y: x|y, [ basecip.graph.nodes[i]['contracted']
                                for i in basecip.core_nodes if 'edge' not in basecip.graph.nodes[i] ] ))

        edges_in_core = [n for n,d in expanded_orig_graph_collapsed.nodes(data=True)
                             if 'edge' in d and all([z in nodes_in_core for z in expanded_orig_graph_collapsed.neighbors(n) ])]
        for n in nodes_in_core[1:]+edges_in_core:
            transformutil.merge_edge(expanded_orig_graph_collapsed, nodes_in_core[0], n)


        # distances...
        dist = nx.single_source_shortest_path_length(expanded_orig_graph_collapsed,
                                        nodes_in_core[0],max(self.decomposition_args['base_thickness_list']))

        # set distance dependant label
        ddl = 'distance_dependent_label'
        for id, dst in dist.items():
            if dst>0:
                expanded_orig_graph.nodes[id][ddl] = expanded_orig_graph.nodes[id]['hlabel'] + dst


        for base_thickness in self.decomposition_args['base_thickness_list']:

            interface_nodes = [id for id, dst in dist.items()
                       if 0 < dst <= base_thickness]
            interface_hash = lsgg_cip.graph_hash(expanded_orig_graph_collapsed.subgraph(interface_nodes))
            cip=copy.deepcopy(basecip)
            cip.interface_nodes=interface_nodes
            cip.interface_graph = expanded_orig_graph.subgraph(interface_nodes).copy()
            cip.core_nodes=nodes_in_core+edges_in_core
            cip.interface_hash =  hash((interface_hash,cip.interface_hash))
            cip.graph= expanded_orig_graph.subgraph(interface_nodes+nodes_in_core+edges_in_core).copy()


            #print cip.interface_hash, cip.core_hash, root_node
            yield cip


def extract_core_and_interface(root_node=None,
                               graph=None,
                               radius=None,
                               thickness=None,
                               thickness_loco=2):
    # MAKE A NORMAL CIP AS IN LSGG_CIP
    graph = lsgg_cip._edge_to_vertex(graph)
    lsgg_cip._add_hlabel(graph)
    dist = {a: b for (a, b) in lsgg_cip.short_paths(graph,
                                                    root_node if isinstance(root_node, list) else [root_node],
                                                    thickness + radius + thickness_loco)}

    core_nodes = [id for id, dst in dist.items() if dst <= radius]
    interface_nodes = [id for id, dst in dist.items()
                       if radius < dst <= radius + thickness]

    normal_cip = lsgg_cip._finalize_cip(root_node, graph, radius, thickness, dist, core_nodes, interface_nodes)

    # NOW COMES THE loco PART

    loco_nodes = [id for id, dst in dist.items()
                  if (radius + thickness) < dst <= (radius + thickness + thickness_loco)]

    loco_graph = graph.subgraph(loco_nodes)

    loosecontext = nx.Graph(loco_graph)
    nn = loosecontext.number_of_nodes() > 2
    # eden doesnt like empty graphs, they should just be a 0 vector...
    normal_cip.loco_vectors = [lsgg_cip.eg.vectorize([loosecontext])] if nn else [None]

    return normal_cip






