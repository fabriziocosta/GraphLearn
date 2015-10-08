from graphlearn.graphlearn import GraphLearnSampler
from graphlearn.graph import core_substitution, graph_clean

from cip_predictor import CipPredictor

import logging
logger = logging.getLogger(__name__)

class GraphLearnSamplerCipML(GraphLearnSampler, CipPredictor):
    """Extends database and predictor with an intelligent graph sampler.
    
    Attributes:
        online_learning             If true cip pairs are stored in database while sampling.
        intelligent_cip_selection   If true cip candidates are sorted by descending score.
    
    Constants:
        MIN_BATCH_SIZE              Minimum number of scores non used before predictor re-fit
        MIN_SCORES_2_PREDICT        Minimum number of scores available before predictor being used
    """
    
    MIN_BATCH_SIZE = 500
    MIN_SCORES_2_PREDICT = 1500
    
    def __init__(self, *args, **kwargs):
        GraphLearnSampler.__init__(self, *args, **kwargs)
        CipPredictor.__init__(self)
        
        self.online_learning = True
        self.intelligent_cip_selection = False


    def _propose_graph(self, graph):
        """Override to store cip pairs to database if online_learning is true."""

        original_cip = self.select_original_cip(graph)
        candidate_cips = self._candidate_cips(original_cip)
        
        for candidate_cip in candidate_cips:
            if candidate_cip.core_hash == original_cip.core_hash:
                continue
            
            graph_new = core_substitution(graph, original_cip.graph, candidate_cip.graph)
            if self.feasibility_checker.check(graph_new):
                graph_clean(graph_new)

                if self.online_learning:
                    self._score(graph)
                    self._score(graph_new)
                    self._new_pair(original_cip, candidate_cip, graph._score, graph_new._score)

                return self.postprocessor.postprocess(graph_new)
            else:
                logger.debug('feasibility checker failed')


    def _candidate_cips(self, original_cip):
        """Candidate cips sorted by descending score if intelligent_cip_selection is true."""
        candidate_cips = list(self._select_cips(original_cip))
        
        if not self.intelligent_cip_selection:
            return candidate_cips
        
        if self.num_scores_fitted < self.MIN_SCORES_2_PREDICT:
            return candidate_cips
        
        databased_cips = [cip for cip in candidate_cips if     self._exist_score(original_cip, cip)]
        predicted_cips = [cip for cip in candidate_cips if not self._exist_score(original_cip, cip)]
        
        databased_cips = self._databased_cips(original_cip, databased_cips)
        predicted_cips = self._predicted_cips(original_cip, predicted_cips)
        
        candidate_cips = databased_cips + predicted_cips
        
        candidate_cips = sorted(candidate_cips, reverse=True)
        
        candidate_cips = [cip for (prediction, cip) in candidate_cips]
        
        return candidate_cips
    
    
    def _sample(self, graph):
        sampled_graph = GraphLearnSampler._sample(self, graph)
        
        if self.num_scores> self.MIN_SCORES_2_PREDICT:
            if self.num_scores - self.num_scores_fitted > self.MIN_BATCH_SIZE:
                self.create_features()
                self.cip_fit()
                
                print("FITTED ", self.num_scores_fitted)
        
        return sampled_graph