import numpy as np

from graphlearn.local_substitution_graph_grammar import LocalSubstitutionGraphGrammar, logger
from graphlearn.util import util
import random
from graphlearn.choice import SelectMax

import logging
logger = logging.getLogger(__name__)

def sample_step(object, transformer, grammar, scorer, selector):
    """
    Parameters
    ----------
    object:
    transformer: test.test(object) -> graph
    grammar: trained on graphs, lsgg.propose(graph) -> objects
    scorer: score.decision_function(object) -> number

    Returns
    -------
        object
    """
    graph = transformer.encode_single(object)  
    util.valid_gl_graph(graph)
    proposal_graphs = list(grammar.neighbors_sample(graph,1))+transformer.decode([graph])
    proposal_objects = list(transformer.decode(proposal_graphs))
    scores = scorer.decision_function(proposal_objects)
    new_object, score = selector.select(proposal_objects, scores)
    return new_object, score


def sample(graph, transformer=None, grammar=None, scorer=None, selector=None, n_steps=75, return_score=False):
    for i in range(n_steps):
        graph, score = sample_step(graph, transformer, grammar, scorer,selector)
    if return_score:
        return graph, score
    return graph


def sample_sizeconstraint(graph,penalty=0.01, **kwargs):
    kwargs['scorer'].sizefactor = len(graph)
    kwargs['scorer'].sizepenalty = penalty
    return sample(graph,**kwargs)




class Sampler(object):
    def __init__(self,**sampleargs):
        self.faster=False
        self.num_sample = 1
        self.__dict__.update(sampleargs)
        self.history=[]
        
    
    def sample_sizeconstraint(self,graph, penalty=0.0):
        self.scorer.sizefactor = len(graph)
        self.scorer.sizepenalty = penalty
        return self.sample(graph)

    def sample_burnin(self,graph):
        res= []
        for i in range(self.n_steps):
            if self.num_sample==1:
                graph, score = self.sample_step(graph,i)
            else:
                graph,score = self.sample_step_multi(graph,i)
            if i >= self.burnin: 
                if (i - self.burnin) % self.emit ==0:
                    res.append(graph)
        return res
            
    def sample(self,graph):
        for i in range(self.n_steps):
            graph, score = self.sample_step(graph,i)
        return graph

    def sample_step(self,object,step):
        if object is None: return None,0
        graph = self.transformer.encode_single(object)
        util.valid_gl_graph(graph)
        if self.faster:
            proposal_graphs = list(self.grammar.neighbors_sample_faster(graph,1))+[graph]
        else:
            proposal_graphs = list(self.grammar.neighbors_sample(graph,1))+[graph]

        proposal_objects = list(self.transformer.decode(proposal_graphs))
        
        if len(proposal_objects) <= 1: 
            logger.log(10,"reached a dead-end graph, attempting to backtrack at step " % step)
            if len(self.history) < 2:
                return None,0
            self.history.pop() # the problematic graph should be on top of the stack
            return self.history.pop() 

        scores = self.scorer.decision_function(proposal_objects)
        obj_score = self.selector.select(proposal_objects, scores)
        self.history.append(obj_score)
        return obj_score

    def sample_step_multi(self,object,step):
        if object is None: return None,0

        # a graph is something that the grammar understands
        graph = self.transformer.encode_single(object)
        util.valid_gl_graph(graph)
        startscore = self.scorer.decision_function([object])[0]
        current = graph, startscore
        scorehist= [] 
        backupmgr = Backupmgr(15)
        for g in self.grammar.neighbors_sample(graph, self.num_sample):
            proposal_object = self.transformer._decode_single(g)
            score = self.scorer.decision_function([proposal_object])[0]
            scorehist.append(score)
            backupmgr.push((score,proposal_object))
            if score > current[1]:
                current = proposal_object,score
        
        if startscore == current[1]:
            score,pobj = backupmgr.get() 
            logger.log(10,"reached a dead-end graph, choose probabilistically at step %d" % step)
            current = pobj,score

        self.history.append(current)
        return current

class Backupmgr():
    def __init__(self, maxsize):
        self.maxsize = maxsize 
        self.data = []
    def push(self,x):
        self.data.append(x)
        self.data.sort(key = lambda x: x[0] )
        self.data=self.data[:self.maxsize]
    def get(self):
        return random.choices(self.data,[ p for p,g in self.data ])[0]


class LocalSubstitutionGraphGrammarSample(LocalSubstitutionGraphGrammar):

    def _sample_size_adjusted(self, subs):
        '''proposals are sampled, such that increasing or decreasing the graph has equal probability'''
        probs= self._make_size_adjusted_probabilities(subs)
        return self._sample(subs, probs)

    def _make_size_adjusted_probabilities(self, subs):

        # get size change for each  substitution
        diffs = [ len(b.core_nodes)-len(a.core_nodes) for a,b in subs  ]

        z = diffs.count(0)
        s = sum(x<0 for x in diffs)
        g = sum(x>0 for x in diffs)
        logger.log(5,"assigned probabilities: same: %d, smaller %d, bigger %d" % (z,s,g))
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

    def _sample(self, subs, probabilities):
        if len(subs)==0: return []
        suu = sum(probabilities)
        p=[x/suu for x in probabilities]
        samples = np.random.choice( list(range(len(subs))) ,
                size=len(subs),
                replace=False,
                p=p)
        return [subs[i] for i in samples[::-1]]

    def neighbors_core(self, graph, core):
        """iterator over all neighbors of graph (that are conceiveable by the grammar)"""

        graph_cip = self._get_cip(core, graph)
        cip_substitutions = [(graph_cip, congruent_cip)
                             for congruent_cip in self._get_congruent_cips(graph_cip)]

        for cip, congruent_cip in  self._sample_size_adjusted(cip_substitutions):
            graph_ = self._substitute_core(graph, cip, congruent_cip)
            if graph_ is not None:
                yield graph_

    def neighbors_sample(self, graph, n_neighbors):
        """neighbors_sample. might be a little bit faster by avoiding cip extractions,
        chooses a node first and then picks form the subs evenly
        """
        cores = list(self._get_cores(graph))
        random.shuffle(cores)
        for core in cores:
            for graph_ in self.neighbors_core(graph, core):
                if n_neighbors > 0:
                    yield graph_
                    n_neighbors = n_neighbors - 1
