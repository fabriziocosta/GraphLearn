from __future__ import division

from numpy import mean
import pickle



class CipDatabase(object):
    """Database with scores ratio between cip pairs."""
    
    def __init__(self):
        self.num_scores = 0
        self.cip_graphs = []
        self.graph2position = {}
        self.cip_relation = {}


    def _new_pair(self, cip_start, cip_end, score_start, score_end):
        """Add cip pair and score to the database."""
        self._new_graph(cip_start)
        self._new_graph(cip_end)
            
        score = score_end / score_start
        
        self._new_score(cip_start, cip_end, score)


    def _new_graph(self, cip):
        """Add new graph to the database."""
        interface = cip.interface_hash
        core = cip.core_hash
        
        if interface not in self.graph2position:
            self.graph2position[interface] = {}
        
        if core in self.graph2position[interface]:
            return
        
        self.graph2position[interface][core] = len(self.cip_graphs)
        
        self.cip_graphs.append(cip.graph)
        
        
    def _new_score(self, cip_start, cip_end, score):
        """Add new score to the database."""
        interface = cip_start.interface_hash
        core_start = cip_start.core_hash
        core_end = cip_end.core_hash
        
        if interface not in self.cip_relation:
            self.cip_relation[interface] = {}
            
        if core_start not in self.cip_relation[interface]:
            self.cip_relation[interface][core_start] = {}
            
        if core_end not in self.cip_relation[interface][core_start]:
            self.cip_relation[interface][core_start][core_end] = []
        
        self.num_scores += 1
        
        self.cip_relation[interface][core_start][core_end].append((score, False))
        
        
    def _exist_score(self, cip_start, cip_end):
        """Check if score exist for a cip pair."""
        interface = cip_start.interface_hash
        core_start = cip_start.core_hash
        core_end = cip_end.core_hash  
        
        if interface not in self.cip_relation:
            return False
        
        if core_start not in self.cip_relation[interface]:
            return False
        
        if core_end not in self.cip_relation[interface][core_start]:
            return False
        
        return True 
    
    
    def _databased_cips(self, original_cip, candidate_cips):
        """Return average scores of a list of candidate cips."""
        interface = original_cip.interface_hash
        core_start = original_cip.core_hash
        
        y = []
        for candidate_cip in candidate_cips:
            core_end = candidate_cip.core_hash 
            score = mean(self.cip_relation[interface][core_start][core_end][0])
            y.append(score)
            
        return zip(y, candidate_cips)
    
    
    def get_items(self):
        """Generator to iterate over the database structure."""
        for interface, cores_start in self.cip_relation.items():
            for core_start, cores_end in cores_start.items():
                for core_end, scores in cores_end.items():
                    yield interface, core_start, core_end, scores[0]
    
    
    def save_cip_data(self):
        """Save database to file."""
        f = open("cip_database.data", "w")
        pickle.dump(self.cip_graphs, f)
        pickle.dump(self.graph2position, f)
        pickle.dump(self.cip_relation, f)
        f.close()


    def load_cip_data(self):
        """Load database from file."""
        f = open("cip_database.data", "r")
        self.cip_graphs = pickle.load(f)
        self.graph2position = pickle.load(f)
        self.cip_relation = pickle.load(f)
        f.close()
