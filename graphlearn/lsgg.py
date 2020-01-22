#!/usr/bin/env python

"""Provides the graph grammar class."""

import random
from collections import defaultdict
from graphlearn import lsgg_cip
import logging
import numpy as np

logger = logging.getLogger(__name__)
from graphlearn.util.multi import mpmap



class lsgg(lsgg_core):
    def set_core_size(self, vals):
        self.decomposition_args['radius_list'] = vals

    def set_context(self, val):
        self.decomposition_args['thickness'] = val

    def set_min_count(self, val):
        self.filter_args['min_interface_count'] = val
        self.filter_args['min_cip_count'] = val

    def get_min_count(self):
        return self.filter_args['min_cip_count']

    def reset_productions(self):
        self.productions = defaultdict(dict)

    def propose(self, graph):
        return list(self.neighbors(graph))

    def is_fit(self):
        return len(self.productions) > 0 

    ########
    # print
    ########
    def size(self):
        """size."""
        n_interfaces = len(self.productions)

        cores = set()
        n_productions = 0
        for interface in self.productions.keys():
            n_productions += len(self.productions[interface]) * (len(self.productions[interface]) - 1)
            for core in self.productions[interface].keys():
                cores.add(core)

        n_cores = len(cores)
        n_cips = sum(len(self.productions[interface])
                     for interface in self.productions)

        return n_interfaces, n_cores, n_cips, n_productions

    def __repr__(self):
        """repr."""
        n_interfaces, n_cores, n_cips, n_productions = self.size()
        txt = '#interfaces: %5d   ' % n_interfaces
        txt += '#cores: %5d   ' % n_cores
        txt += '#core-interface-pairs: %5d  ' % n_cips
        txt += '#production-rules: %5d' % n_productions
        return txt

    def fit(self, graphs, n_jobs=1):
        if n_jobs==1:
            super().fit(graphs)

        res = mpmap(self._list_cip_extraction, graphs, poolsize=n_jobs)
        for ciplist in res:
            for cip in ciplist: 
                if len(cip.interface_nodes) > 0:
                    self._add_cip(cip)

        self._add_library(graphs)
        self._cip_frequency_filter()
        self._is_fit = True
        return self

    def _list_cip_extraction(self,graph):
        return list(self._graph_to_cips(graph))


class lsgg_sample(lsgg):

    def _neighbors_sample_order_proposals(self,subs):
        probs= self.get_size_proba(subs)
        return self.order_proba(subs,probs)

    def get_size_proba(self,subs):
        diffs = [ b.core_nodes_count-a.core_nodes_count for a,b in subs  ]
        z = diffs.count(0) 
        s = sum(x<0 for x in diffs)
        g = sum(x>0 for x in diffs)
        logger.log(5,f"assigned probabilities: same:{z}, smaller{s}, bigger{g}")
        z = 1/z if z !=0 else 1
        s = 1/s if s !=0 else 1
        g = 1/g if g !=0 else 1
        def f(d):
            if d ==0:
                return z 
            if d>0:
                return g
            return s
        return [ f(d)  for d in diffs]
    
    def order_proba(self,subs, probabilities):
        if len(subs)==0: return []
        
        suu = sum(probabilities)
        p=[x/suu for x in probabilities]
        samples = np.random.choice( list(range(len(subs))) ,
                size=len(subs),
                replace=False,
                p=p)
        return [subs[i] for i in samples[::-1]]

    def neighbors_sample(self, graph, n_neighbors,shuffle_accurate=True):
        """neighbors_sample. samples from all possible replacements"""

        cips = self._graph_to_cips(graph)
        subs = [ (cip,con_cip) for cip in cips
                               for con_cip in self._congruent_cips(cip)]
        
        if len(subs) < 1: logger.info('no congruent cips')
        
        if shuffle_accurate: subs = self._neighbors_sample_order_proposals(subs)
        else:  random.shuffle(subs)

        n_neighbors_counter = n_neighbors
        while subs:
            cip,cip_ = subs.pop()
            #print(cip,cip_) # todo clean this up 
            graph_ = self._core_substitution(graph, cip, cip_)
            if graph_ is not None:
                if n_neighbors_counter > 0:
                    n_neighbors_counter = n_neighbors_counter - 1
                    yield graph_
                else:
                    return
        logger.info("neighbors_sample sampled few graphs")

    def neighbors_sample_faster(self, graph, n_neighbors):
        """neighbors_sample. might be a little bit faster by avoiding cip extractions,
        chooses a node first and then picks form the subs evenly
        """
        n_neighbors_counter = n_neighbors
        sanity = max(n_neighbors*3, 15) 
        mycips = {}
        while sanity: 

            # select a cip: 
            sanity -=1
            rootradthi = (random.choice(list(graph)), 
                random.choice( self.decomposition_args['radius_list'] ),
                self.decomposition_args['thickness'])

            if rootradthi in mycips:
                cip = mycips[rootradthi]
            else:
                root,rad, thi = rootradthi
                cip= self._extract_cip(root_node=root,
                        graph=graph,
                        radius=rad*2,
                        thickness=thi*2)
                mycips[rootradthi] = cip


            if type(cip) != lsgg_cip.CoreInterfacePair:
                logger.log(5,'0 cips extracted')
                continue

            # select cip to substitute: 
            subs = [(cip,congru) for congru in self._congruent_cips(cip) ]
            if len(subs)==0: 
                logger.log(5,'no congruent cips')
                continue
            subs = self._neighbors_sample_order_proposals(subs)
            cip, cip_ = subs[0]
            
            # substitute
            graph_ = self._core_substitution(graph, cip, cip_)
            if graph_ is not None:
                if n_neighbors_counter > 0:
                    n_neighbors_counter = n_neighbors_counter - 1
                    yield graph_
                else:
                    return
        logger.info("neighbors_sample_faster sampled few graphs")





class lsgg_core(object):

    def __init__(self,
                 decomposition_args={"radius_list": [0, 1],
                                     "thickness": 1},
                 filter_args={"min_cip_count": 2,
                              "min_interface_count": 2},
                 cip_root_all=False,
                 double_radius_and_thickness=True
                 ):
        """Parameters
        ----------
        decomposition_args:
        filter_args
        cip_root_all : include edges as possible roots
        double_decomp_args: interpret options for radius and thickness 
                as half step (default is full step)
        """
        self.productions = defaultdict(dict)
        self.decomposition_args = decomposition_args
        self.filter_args = filter_args
        self.cip_root_all = cip_root_all
        if  double_radius_and_thickness:
            self.double_radius_and_thickness()

    def double_radius_and_thickness(self):
            self.decomposition_args['radius_list'] = [i*2 for i in  self.decomposition_args['radius_list']]
            self.decomposition_args['thickness'] = 2 * self.decomposition_args['thickness']



    ###########
    # FITTING
    ##########
    def fit(self, graphs):
        self._add_library(graphs)
        self._cip_frequency_filter()
        return self
    

    def _add_library(self,graphs):
        for graph in graphs:
            self._add_graph(graph)

    def _add_graph(self, graph):
        """see fit"""
        for cip in self._graph_to_cips(graph):
            if len(cip.interface_nodes) > 0:
                self._add_cip(cip)


    def _graph_to_cips(self, graph):
        """see fit"""
        thickness = self.decomposition_args['thickness']
        for root in self._roots(graph):
            for radius in self.decomposition_args['radius_list']:
                x= self._extract_cip(root_node=root,
                                                   graph=graph,
                                                   radius=radius,
                                                   thickness=thickness)
                if x:
                    yield x


    def _extract_cip(self, **kwargs):
        return lsgg_cip.extract_cip(**kwargs)


    def _add_cip(self, cip):
        """see fit"""
        # setdefault is a fun function
        self.productions[cip.interface_hash].setdefault(cip.core_hash, cip).count += 1

    def _cip_frequency_filter(self):
        logger.log(10,"grammar bevore freq filter: %s" % str(self))
        """Remove infrequent cores and interfaces. see fit"""
        min_cip = self.filter_args['min_cip_count']
        min_inter = self.filter_args['min_interface_count']
        for interface in list(self.productions.keys()):
            for core in list(self.productions[interface].keys()):
                if self.productions[interface][core].count < min_cip:
                    self.productions[interface].pop(core)
            if len(self.productions[interface]) < min_inter:
                self.productions.pop(interface)
        logger.log(10, self)

    ##############
    #  APPLYING A PRODUCTION
    #############
    def _congruent_cips(self, cip):
        """all cips in the grammar that are congruent to cip in random order.
        congruent means they have the same interface-hash-value"""
        cips = self.productions.get(cip.interface_hash, {}).values()
        cips_ = [cip_ for cip_ in cips if cip_.core_hash != cip.core_hash]
        random.shuffle(cips_)
        return cips_

    def _core_substitution(self, graph, cip, cip_):
        try:
            return lsgg_cip.core_substitution(graph, cip, cip_)
        except:
            print("core sub failed (continuing anyway):")
            import structout as so
            so.gprint([graph, cip.graph, cip_.graph],color =[[[],[]]]+
                    [ [c.interface_nodes, c.core_nodes]  for c in [cip,cip_]])
            return None

    def _neighbors_given_cips(self, graph, orig_cips):
        """iterator over graphs generted by substituting all orig_cips in graph (with cips from grammar)"""
        for cip in orig_cips:
            cips_ = self._congruent_cips(cip)
            for cip_ in cips_:
                graph_ = self._core_substitution(graph, cip, cip_)
                if graph_ is not None:
                    yield graph_

    def neighbors(self, graph):
        """iterator over all neighbors of graph (that are conceiveable by the grammar)"""
        cips = self._graph_to_cips(graph)
        it = self._neighbors_given_cips(graph, cips)
        for neighbor in it:
            yield neighbor

    def _roots(self, graph):
        '''option to choose edge nodes as root'''
        if self.cip_root_all:
            graph = lsgg_cip._edge_to_vertex(graph)
        return graph.nodes()

  
    def __repr__(self):
        return f"interfaces {len(self.productions)} cores: {len(set([ i.core_hash for v in self.productions.values() for i in v ]))}"
    
    
    
    

