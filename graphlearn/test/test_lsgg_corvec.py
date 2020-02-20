#!/usr/bin/env python

"""Provides the graph grammar class."""

#from graphlearn import lsgg_core_interface_pair as lcip
#from graphlearn import local_substitution_graph_grammar as lsgg


from graphlearn.util import util
from ego.decomposition.paired_neighborhoods import decompose_neighborhood
import networkx as nx

import logging
logger = logging.getLogger(__name__)

def test_cove():
    from graphlearn import cipcorevector as ccv 

    lsgg = ccv.LsggCoreVec(decompose_neighborhood)
    g = nx.path_graph(5)
    g.add_edge(2,4)
    g = util._edenize_for_testing(g)

    cores = lsgg._get_cores(g)
    cip = lsgg._make_cip(cores[0], g)

    assert cip.core_vec.sum()==2 



