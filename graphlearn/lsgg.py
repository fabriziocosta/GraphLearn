#!/usr/bin/env python

"""Provides the graph grammar class."""

import random
from collections import defaultdict
from graphlearn import lsgg_cip
import logging
import numpy as np

logger = logging.getLogger(__name__)
from graphlearn.util.multi import mpmap

class lsgg(object):
    """Graph grammar."""

    def __init__(self,
                 decomposition_args={"radius_list": [0, 1],
                                     "thickness_list": [1, 2]},
                 filter_args={"min_cip_count": 2,
                              "min_interface_count": 2},
                 cip_root_all=False,
                 half_step_distance=False,
                 maxgrowth = 999
                 ):
        """
        Init.

        Parameters
        ----------
        decomposition_args:
        filter_args
        cip_root_all : include edges as possible roots
        half_step_distance: interpret options for radius and thickness as half step (default is full step)
        """
        self.productions = defaultdict(dict)
        self.decomposition_args = decomposition_args
        self.filter_args = filter_args
        self.cip_root_all = cip_root_all
        self.half_step_distance = half_step_distance
        self.maxgrowth = maxgrowth

    def set_core_size(self, vals):
        self.decomposition_args['radius_list'] = vals

    def set_context(self, val):
        self.decomposition_args['thickness_list'] = [val]

    def set_min_count(self, val):
        self.filter_args['min_interface_count'] = val
        self.filter_args['min_cip_count'] = val

    def get_min_count(self):
        return self.filter_args['min_cip_count']

    def reset_productions(self):
        self.productions = defaultdict(dict)

    ###########
    # FITTING
    ##########
    def fit(self, graphs, n_jobs=1):
        """fit.
            _add_production will extract all CIPS
            _add_cip will add to the production dictionary self.productions[interfacehash][corehash]=cip
            _cip_frequency filter applies the filter_args that are set in __init__
        """
        if n_jobs==1:
            for graph in graphs:
                self._add_productions(graph)
        else: 
            res = mpmap(self.my_cip_extraction, graphs, poolsize=n_jobs)
            for ciplist in res:
                for cip in ciplist: 
                    if len(cip.interface_nodes) > 0:
                        self._add_cip(cip)
        logger.log(5, self)
        self._cip_frequency_filter()
        logger.log(5, self)
        self._is_fit = True
        return self

    def _add_productions(self, graph):
        """see fit"""
        for cip in self._cip_extraction(graph):
            if len(cip.interface_nodes) > 0:
                self._add_cip(cip)

    def is_fit(self):
        return self._is_fit



    def _cip_extraction(self, graph):
        """see fit"""
        for root in self._roots(graph):
            for cip in self._cip_extraction_given_root(graph, root):
                yield cip

    def my_cip_extraction(self,graph):
        return list(self._cip_extraction(graph))

    def _extract_core_and_interface(self, **kwargs):
        return lsgg_cip.extract_core_and_interface(**kwargs)

    def _cip_extraction_given_root(self, graph, root):
        """helper of _cip_extraction. See fit"""
        for radius in self.decomposition_args['radius_list']:
            if not self.half_step_distance:
                radius = radius * 2
            for thickness in self.decomposition_args['thickness_list']:
                if not self.half_step_distance:
                    thickness = thickness * 2
                x= self._extract_core_and_interface(root_node=root,
                                                       graph=graph,
                                                       radius=radius,
                                                       thickness=thickness)
                if x:
                    yield x

    def _add_cip(self, cip):
        """see fit"""
        # setdefault is a fun function
        self.productions[cip.interface_hash].setdefault(cip.core_hash, cip).count += 1

    def _cip_frequency_filter(self):
        """Remove infrequent cores and interfaces. see fit"""
        min_cip = self.filter_args['min_cip_count']
        min_inter = self.filter_args['min_interface_count']
        for interface in list(self.productions.keys()):
            for core in list(self.productions[interface].keys()):
                if self.productions[interface][core].count < min_cip:
                    self.productions[interface].pop(core)
            if len(self.productions[interface]) < min_inter:
                self.productions.pop(interface)

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
        cips = self._cip_extraction(graph)
        it = self._neighbors_given_cips(graph, cips)
        for neighbor in it:
            yield neighbor


    def _neighbors_sample_order_proposals(self,subs):
        #random.shuffle(subs)
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

    def neighbors_sample(self, graph, n_neighbors):
        """neighbors_sample. samples from all possible replacements"""

        cips = self._cip_extraction(graph)
        subs = [ (cip,con_cip) for cip in cips
                               for con_cip in self._congruent_cips(cip)]
        
        if len(subs) < 1: logger.info('no congruent cips')
        
        subs = self._neighbors_sample_order_proposals(subs)
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
                random.choice( self.decomposition_args['thickness_list'] ))
            if rootradthi in mycips:
                cip = mycips[rootradthi]
            else:
                root,rad, thi = rootradthi
                cip= self._extract_core_and_interface(root_node=root,
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

    def propose(self, graph):
        return list(self.neighbors(graph))

    def _roots(self, graph):
        '''option to choose edge nodes as root'''
        if self.cip_root_all:
            graph = lsgg_cip._edge_to_vertex(graph)
        return graph.nodes()

    ########
    # why is this here? is this a copy of the thing in utils.py?
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



def internal_tesst_paral():
    # python -c "import lsgg as s; s.internal_tesst_paral()"
    # lets get sum data
    from toolz import curry, pipe
    from eden_chem.io.pubchem import download
    from eden_chem.io.rdkitutils import sdf_to_nx
    download_active = curry(download)(active=True)
    download_inactive = curry(download)(active=False)
    def get_pos_graphs(assay_id): return pipe(assay_id, download_active, sdf_to_nx, list)
    assay_id='624249'
    gr = get_pos_graphs(assay_id)
    print ("grlen",len(gr))
    gramm= lsgg().fit(gr[:100]) 
    print ("fit")
    import basics as b  # from my own repo :)
    b.mpmap( internal_f , [(gramm,g) for g in gr[:100]])

def internal_f(g):
    return list(g[0].neighbors(g[1]))



    
    
    
    

