#!/usr/bin/env python

"""Provides the graph grammar class."""

from graphlearn3 import lsgg_cip
import logging
from graphlearn3.util import util
import networkx as nx

logger = logging.getLogger(__name__)


def test_fit():
    lsggg = util.test_get_grammar()
    assert (4 == sum(len(e) for e in lsggg.productions.values()))
    # gprint( [e.graph for e in lsggg.productions[49532].values() ])
    # gprint( [e.graph for e in lsggg.productions[29902].values() ])


def test_extract_core_and_interface():
    graph = nx.path_graph(4)
    util._edenize_for_testing(graph)
    res = lsgg_cip.extract_core_and_interface(root_node=3, graph=graph, radius=1, thickness=1)
    # gprint(res.graph)
    assert ('cor' in str(res))


def test_neighbors():
    # make a grammar
    lsgg = util.test_get_grammar()

    # make agraph
    g = nx.path_graph(4)
    g = util._edenize_for_testing(g)
    g.node[3]['label'] = '5'
    stuff = list(lsgg.neighbors(g))
    assert (6 == len(stuff))


def test_some_neighbors():
    # make a grammar
    lsgg = util.test_get_grammar()
    # make agraph
    g = nx.path_graph(4)
    g = util._edenize_for_testing(g)
    g.node[3]['label'] = '5'
    assert (1 == len(list(lsgg.neighbors_sample(g, 1))))
    assert (2 == len(list(lsgg.neighbors_sample(g, 2))))
    assert (3 == len(list(lsgg.neighbors_sample(g, 3))))
    # gprint(list( lsgg.some_neighbors(g,1) ))
    # gprint(list( lsgg.some_neighbors(g,2) ))
    # gprint(list( lsgg.some_neighbors(g,3) ))
