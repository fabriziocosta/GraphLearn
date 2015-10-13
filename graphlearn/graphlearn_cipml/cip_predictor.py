from numpy import median
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model.stochastic_gradient import SGDRegressor

from cip_database import CipDatabase



class CipPredictor(CipDatabase):
    """Extend the database with a predictor."""
    
    def __init__(self):
        CipDatabase.__init__(self)
        
        self.num_scores_fitted = 0
        self.X = []
        self.y = []

        #self.predictor = SGDRegressor()
        self.predictor = KNeighborsRegressor()
    
    
    def create_features(self):
        """Create feature vectors between database pairs."""
        self.X = []
        self.y = []
        
        self.cip_fvs = self.vectorizer.transform_single(self.cip_graphs)
        
        for interface, core_start, core_end, scores in self.get_items():
            pos_start = self.graph2position[interface][core_start]
            pos_end = self.graph2position[interface][core_end]    
                        
            vector_start = self.cip_fvs[pos_start]
            vector_end = self.cip_fvs[pos_end]

            feature_vector = vector_start - vector_end

            if len(feature_vector.data):
                score = median(scores)
                max_drift = max(score - min(scores), max(scores) - score) 
                        
                self.X.append(feature_vector)
                self.y.append(score)
      
        self.X = vstack(self.X)
        
        
    def cip_fit(self):
        """Fit the predictor."""
        
        self.predictor.fit(self.X, self.y)
        self.num_scores_fitted = self.num_scores
        

    def _predicted_cips(self, original_cip, candidate_cips):
        """Return average scores of a list of candidate cips."""
        original_fv = self.vectorizer.transform_single(original_cip.graph)
        original_fv = original_fv[0]
            
        candidate_graphs = [candidate_cip.graph for candidate_cip in candidate_cips]
        candidate_fvs = self.vectorizer.transform_single(candidate_graphs)
            
        pairwise_fvs = [original_fv - candidate_fv for candidate_fv in candidate_fvs]
        pairwise_fvs = vstack(pairwise_fvs)

        y = self.predictor.predict(pairwise_fvs)

        return zip(y, candidate_cips)
    
    
    def save_cip_data(self):
        """Save database and feature vectors to files."""
        CipDatabase.save_cip_data(self)
        
        dump_svmlight_file(self.X, self.y, 'cip_rank_regression.data', zero_based=False)


    def load_cip_data(self):
        """ Load database and feature vectors from files."""
        CipDatabase.load_cip_data(self)
        self.X, self.y = load_svmlight_file('cip_rank_regression.data', n_features=self.vectorizer.feature_size, zero_based=False)