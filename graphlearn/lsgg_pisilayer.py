

from graphlearn.test import transformutil
import copy
import functools
from graphlearn import local_substitution_graph_grammar
from graphlearn import lsgg_core_interface_pair
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import logging
logger = logging.getLogger(__name__)
from graphlearn import lsgg_layered 
from graphlearn import lsgg_pisi


class lsgg_pisilayer( lsgg_layered.lsgg_layered ,lsgg_pisi.PiSi):





    def __init__(self, **kwargs):
        super(lsgg_pisilayer,self).__init__(**kwargs)



 
    def _make_base_cip(self,graph,core):
        exp_base_graph = lsgg_core_interface_pair._edge_to_vertex(graph.graph['original'])
        base_core = self._make_base_core(exp_base_graph, core)
        if len(base_core) == len(exp_base_graph):
            logger.log(10, 'core as big as graph -> no interface ->  return None')
            return None
        return  lsgg_pisi.CIP_PiSi(core=base_core, graph=exp_base_graph,
                                      thickness=self.base_thickness,
                                      thickness_pisi = self.thickness_pisi)


