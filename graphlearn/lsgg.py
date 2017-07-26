#!/usr/bin/env python

"""Provides the graph grammar class."""

import random
from collections import defaultdict
from lsgg_compose_util import extract_core_and_interface, core_substitution
import logging
logger = logging.getLogger(__name__)

_hash_bitmask_ = 2**16 - 1


class lsgg(object):
    """Graph grammar."""

    def __init__(self,
                 decomposition_args={"radius_list": [0, 1],
                                     "thickness_list": [1, 2],
                                     'hash_bitmask': _hash_bitmask_},
                 filter_args={"min_cip_count": 2,
                              "min_interface_count": 2}
                 ):
        """
        Init.

        Parameters
        ----------
        decomposition_args:
        filter_args
        """
        self.productions = defaultdict(dict)
        self.decomposition_args = decomposition_args
        self.filter_args = filter_args

    def fit(self, graphs):
        """fit."""
        for graph in graphs:
            self._add_productions(graph)
        self._cip_frequency_filter()

    def _add_productions(self, graph):
        for cip in self._cip_extraction(graph):
            self._add_cip(cip)

    def _cip_extraction_given_root(self, graph, root):
        hash_bitmask = self.decomposition_args['hash_bitmask']
        for radius in self.decomposition_args['radius_list']:
            radius = radius * 2
            for thickness in self.decomposition_args['thickness_list']:
                thickness = thickness * 2
                yield extract_core_and_interface(root_node=root,
                                                 graph=graph,
                                                 radius=radius,
                                                 thickness=thickness,
                                                 hash_bitmask=hash_bitmask)

    def _cip_extraction(self, graph):
        for root in graph.nodes():
            for cip in self._cip_extraction_given_root(graph, root):
                yield cip

    def _add_cip(self, cip):
        # setdefault is a fun function
        self.productions[cip.interface_hash].setdefault(cip.core_hash, cip).count += 1

    def _cip_frequency_filter(self):
        """Remove infrequent cores and interfaces."""
        min_cip = self.filter_args['min_cip_count']
        min_inter = self.filter_args['min_interface_count']
        for interface in self.productions.keys():
            for core in self.productions[interface].keys():
                if self.productions[interface][core].count < min_cip:
                    self.productions[interface].pop(core)
            if len(self.productions[interface]) < min_inter:
                self.productions.pop(interface)

    def _same_interface_cips(self, graph, cip):
        cips = self.productions[cip.interface_hash].values()
        cips_ = [cip_ for cip_ in cips if cip_.core_hash != cip.core_hash]
        random.shuffle(cips_)
        return cips_

    def _neighbors_given_cips(self, graph, cips):
        for cip in cips:
            cips_ = self._same_interface_cips(graph, cip)
            for cip_ in cips_:
                graph_ = core_substitution(graph, cip, cip_)
                if graph_ is not None:
                    yield graph_

    def neighbors(self, graph):
        """neighbors."""
        cips = self._cip_extraction(graph)
        it = self._neighbors_given_cips(graph, cips)
        for neighbor in it:
            yield neighbor

    def neighbors_sample(self, graph, n_neighbors):
        """neighbors_sample."""
        n_neighbors_counter = n_neighbors
        nodes = graph.nodes()
        random.shuffle(nodes)
        for root in nodes:
            cips = self._cip_extraction_given_root(graph, root)
            for neighbor in self._neighbors_given_cips(graph, cips):
                if n_neighbors_counter > 0:
                    n_neighbors_counter = n_neighbors_counter - 1
                    yield neighbor
                else:
                    raise StopIteration

    def size(self):
        """size."""
        n_interfaces = len(self.productions)
        cores = set()
        for interface in self.productions.keys():
            for core in self.productions[interface].keys():
                cores.add(core)
        n_cores = len(cores)
        n_cips = sum(len(self.productions[interface])
                     for interface in self.productions)
        return n_interfaces, n_cores, n_cips
