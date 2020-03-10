import graphlearn.sample
from graphlearn import local_substitution_graph_grammar
from graphlearn import lsgg_core_interface_pair
from graphlearn.test import transformutil
import networkx as nx
import copy
import functools
import logging
logger = logging.getLogger(__name__)



class lsgg_layered(graphlearn.sample.LocalSubstitutionGraphGrammarSample):

    def _substitute_core(self, graph, cip, cip_):
        return lsgg_core_interface_pair.substitute_core(graph.graph['original'], cip, cip_)


    def __init__(self,base_thickness=2,**kwargs):
        if kwargs.get("nodelevel_radius_and_thickness",True):
            self.base_thickness = base_thickness*2
        else:
            self.base_thickness =  base_thickness
        super(lsgg_layered,self).__init__(**kwargs)


    def _get_cip(self, core=None, graph=None):

        """
        whats happening here:
        graph: abstracted
        graph[orig] is the original

        get 2 cips -> make the lower the cip, and hash the interface hashes together -> done
        """

        coarse_cip = lsgg_core_interface_pair.CoreInterfacePair(core=core,
                                                             graph=graph,
                                                             thickness=self.thickness)
        base_cip = self._make_base_cip(graph, core)
        if not base_cip:
            return None


        base_cip.interface_hash = hash((base_cip.interface_hash,coarse_cip.interface_hash))
        return base_cip


    def _make_base_core(self,exp_base_graph, core):
        nodes_in_core = [x for n in core.nodes() for x in core.nodes[n].get('contracted',[]) ]
        edges_in_core = [n for n,d in exp_base_graph.nodes(data=True)
                        if 'edge' in d and all([z in nodes_in_core for z in exp_base_graph.neighbors(n) ])]
        base_core = exp_base_graph.subgraph(nodes_in_core+edges_in_core)
        return base_core

    def _make_base_cip(self,graph,core):
        exp_base_graph = lsgg_core_interface_pair._edge_to_vertex(graph.graph['original'])
        base_core = self._make_base_core(exp_base_graph, core)
        return  lsgg_core_interface_pair.CoreInterfacePair(core=base_core,
                                                              graph=exp_base_graph,
                                                              thickness=self.base_thickness)


def test_lsgg_layered():
    from graphlearn.util import util as util_top
    from graphlearn.test import cycler
    decomposition_args={ "base_thickness":2,
                        "radius_list": [0],
                        "thickness": 1}

    lsggg = lsgg_layered(decomposition_args=decomposition_args)
    g = util_top.test_get_circular_graph()
    gplus=g.copy()
    gplus.nodes[0]['label']='weird'
    c=cycler.Cycler()
    g=c.encode_single(g)
    gplus = c.encode_single(gplus)
    assert lsggg.fit([g, gplus, g,gplus])
    assert len(lsggg.neighbors(gplus).__next__()) == 8








