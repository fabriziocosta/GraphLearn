import graphlearn.sample
from graphlearn import local_substitution_graph_grammar
from graphlearn import lsgg_core_interface_pair as CIP
import networkx as nx
import numpy as np
import random
import  scipy.sparse as sparse
import logging
logger = logging.getLogger(__name__)
from sklearn.metrics.pairwise import cosine_similarity


class CIP_PiSi(CIP.CoreInterfacePair):

    def __init__(self, core, graph, thickness, thickness_pisi):
        '''
        # preprocess, distances of core neighborhood, init counter
        graph = CIP._edge_to_vertex(graph)
        CIP._add_hlabel(graph)
        CIP._add_hlabel(core)
        dist = {a: b for (a, b) in CIP.short_paths(graph, core.nodes(), thickness_pisi)}
        self.count=0

        # core
        self.core_hash = CIP.graph_hash(core)
        self.core_nodes = core.nodes()

        # interface
        self.interface = graph.subgraph([id for id, dst in dist.items() if 0 < dst <= thickness])
        get_node_label = lambda id, node: node[self.ddl]
        self.interface_hash = CIP.graph_hash(self.interface, get_node_label=get_node_label)

        # cip
        self.graph = self._get_cip_graph(self.interface, core, graph, dist)
        '''

        # normal init
        exgraph, dist = self.initialize_params(core, graph, thickness_pisi)
        self.core_hash = CIP.graph_hash(core)
        self.core_nodes = list(core.nodes())
        self.graph = exgraph.subgraph([id for id, dst in dist.items() if dst <= thickness])
        self.interface, self.interface_hash = self.make_interface(exgraph, dist)

        # PISI Stuff
        loosecontext = exgraph.subgraph([i for i,d in dist.items() if 0 < d < thickness_pisi])
        self.pisi_hash = {CIP.graph_hash(loosecontext)}
        self.pisi_vectors = CIP.eg.vectorize([loosecontext])



class PiSi(graphlearn.sample.LocalSubstitutionGraphGrammarSample):


    def __init__(self,thickness_pisi, **kwargs):
        super(PiSi,self).__init__(**kwargs)
        self.thickness_pisi = thickness_pisi*2

    def _get_cip(self, core=None, graph=None):
        return CIP_PiSi( core=core, graph=graph,thickness=self.thickness,  thickness_pisi=self.thickness_pisi)
    
    def _get_congruent_cips(self, cip):
        cips = self.productions.get(cip.interface_hash, {}).values()
        if len(cips) == 0:
            logger.log(10,"no congruent cip in grammar")
        else: 
            cips_ = [(cip_,np.max( cip_.pisi_vectors.dot(cip.pisi_vectors.toarray()[0])))
                         for cip_ in cips if cip_.core_hash != cip.core_hash]
             
            cips_ = [ (a,b) for a,b in cips_ if b > 0]

            if len(cips_) == 0: logger.log(10,"0 cips with pisi-similarity > 0")
            for cip_, di in cips_:
                cip_.pisisimilarity=di
                yield cip_
    
    def _sample_size_adjusted(self, subs):
        sim = [c[1].pisisimilarity for c in subs ]
        logger.log(10, "pisi similarities: "+str(sim))
        p_size = [a * b for a,b in zip (self._make_size_adjusted_probabilities(subs), sim)]
        return self._sample(subs, p_size)


    def _store_cip(self, cip):
                    
        grammarcip = self.productions[cip.interface_hash].setdefault(cip.core_hash, cip)
        grammarcip.count+=1
        if not grammarcip.pisi_hash.intersection(cip.pisi_hash): 
            grammarcip.pisi_vectors= sparse.vstack( (grammarcip.pisi_vectors,  cip.pisi_vectors))
            grammarcip.pisi_hash= grammarcip.pisi_hash.union(cip.pisi_hash)

    def __repr__(self):
        """repr."""
        n_interfaces, n_cores, n_cips, n_productions = self.size()
        txt = '#interfaces: %5d   ' % n_interfaces
        txt += '#cores: %5d   ' % n_cores
        txt += '#core-interface-pairs: %5d   ' % n_cips
        txt += '#production-rules: %5d   ' % n_productions
        txt += '#pisi vectors: %5d   ' % len(set().union(*[ cip.pisi_hash for v in self.productions.values() for cip in v.values()   ]))
        txt += '#count sum: %5d   ' % sum([ cip.count for v in self.productions.values() for cip in v.values()   ])
        return txt




