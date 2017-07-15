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
                 decomposition_args={"radius_list": [0, 2],
                                     "thickness_list": [2, 4],
                                     'hash_bitmask': _hash_bitmask_},
                 filter_args={"min_cip_count": 2,
                              "min_interface_count": 2}
                 ):
        '''
        Parameters
        ----------
        decomposition_args: remember that radius and distance need to be
                            multiplied by 2
        filter_args
        '''
        self.productions = defaultdict(dict)
        self.decomposition_args = decomposition_args
        self.filter_args = filter_args

    def fit(self, graphs):
        self._add(graphs)
        self._filter()

    def _add(self, graphs):
        for g in graphs:
            for cip in self._decompose(g):
                self._production_add_cip(cip)

    def _rooted_decompose(self, graph, root):
        hash_bitmask = self.decomposition_args['hash_bitmask']
        for radius in self.decomposition_args['radius_list']:
            for thickness in self.decomposition_args['thickness_list']:
                yield extract_core_and_interface(root_node=root,
                                                 graph=graph,
                                                 radius=radius,
                                                 thickness=thickness,
                                                 hash_bitmask=hash_bitmask)

    def _decompose(self, graph):
        for root in graph.nodes():
            for e in self._rooted_decompose(graph, root):
                yield e

    def _production_add_cip(self, cip):
        # setdefault is a fun function
        self.productions[cip.interface_hash].setdefault(cip.core_hash, cip).count += 1

    def _filter(self):
        '''
        removes cores that have not been seen often enough
        removes interfaces that have too few cores
        '''
        min_cip = self.filter_args['min_cip_count']
        min_inter = self.filter_args['min_interface_count']
        for interface in self.productions.keys():
            for core in self.productions[interface].keys():
                if self.productions[interface][core].count < min_cip:
                    self.productions[interface].pop(core)
            if len(self.productions[interface]) < min_inter:
                self.productions.pop(interface)


    def _suggest_new_cips(self, graph, orig_cip):
        v = [e for e in self.productions[orig_cip.interface_hash].values()
             if e.core_hash != orig_cip.core_hash]
        random.shuffle(v)
        return v

    def _neighbors_given_orig_cips(self, graph, original_cips):
        for orig in original_cips:
            candidates_new = self._suggest_new_cips(graph, orig)
            for new in candidates_new:
                r = core_substitution(graph, orig, new)
                (yield r) if r else logger.log(5, 'lsgg: a substitution returned None')

    def neighbors(self, graph):
        it = self._neighbors_given_orig_cips(graph, self._decompose(graph))
        for e in it:
            yield e
