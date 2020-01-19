from graphlearn.util import util
import random
import warnings
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

class sampler(object):
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
            warnings.warn(f"reached a dead-end graph, attempting to backtrack at step {step}")
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
        for g in self.grammar.neighbors_sample(graph,self.num_sample,shuffle_accurate=False):
            proposal_object = self.transformer._decode_single(g)
            score = self.scorer.decision_function([proposal_object])[0]
            scorehist.append(score)
            backupmgr.push((score,proposal_object))
            if score > current[1]:
                current = proposal_object,score
            
        
        if startscore == current[1]:
            score,pobj = backupmgr.get() 
            warnings.warn(f"reached a dead-end graph, choose probabilistically at step {step}")
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

def fit():
    pass

def optimize():
    pass




def test_sample_step():
    from graphlearn.score import SimpleDistanceEstimator as SDE
    import networkx as nx
    from graphlearn import choice
    import graphlearn.test.transformutil as transformutil

    lsgg = util.test_get_grammar()
    graph = util._edenize_for_testing(nx.path_graph(4))
    graph.nodes[3]['label'] = '5'
    score_estimator = SDE().fit(util._edenize_for_testing(nx.path_graph(4)))
    graph,score= sample(graph, transformutil.no_transform(), lsgg, score_estimator, choice.SelectMax(), n_steps=2, return_score=True)

    assert (0.000001 > abs(0.319274373045 - score)), score
    print("sambledestdone")

def multi_sample_step(objects, transformer, grammar, scorer, selector, n_neighbors):
    graphs = list(transformer.encode(objects))
    [util.valid_gl_graph(graph) for graph in graphs]
    proposal_graphs = [ prop for graph in graphs for prop in grammar.neighbors_sample_faster(graph,n_neighbors) ]+graphs
    proposal_objects = list(transformer.decode(proposal_graphs))
    scores = scorer.decision_function(proposal_objects)
    objects, scores = selector.select(proposal_objects, scores)
    return objects,scores


def multi_sample(graph, transformer=None, grammar=None, scorer=None, selector=None, n_steps=10, n_neighbors=20):

    # tests are basically in  long_range_graphlearn/code/chem
    graphs=[graph]
    for i in range(n_steps):
        graphs, scores= multi_sample_step(graphs, transformer, grammar, scorer,selector,n_neighbors)
    s=SelectMax()
    return s.select(graphs,scores)[0]
