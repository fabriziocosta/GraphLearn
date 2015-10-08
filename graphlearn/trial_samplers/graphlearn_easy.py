import networkx as nx
import itertools
import random
import processing
import estimator_wrapper
from graphlearn.graph import extract_core_and_interface, core_substitution, graph_clean
from graphlearn.feasibility import FeasibilityChecker
from graphlearn.localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar
from multiprocessing import Pool
import dill
import traceback
from eden import grouper
from eden.graph import Vectorizer
from eden.util import serialize_dict
import logging
from utils import draw
logger = logging.getLogger(__name__)


'''
This is a minimal version of the graph sampler, i tried to remove as much as possible so one can more easily see how it works.

I put these comment blocks to discuss the things that are happening below.

WATCH OUT FOR UPERCASE COMMENTS

'''



class GraphLearnSampler(object):


    '''
    HERE PREPARATIONS FOR SAMPLING ARE TAKEN CARE OF
    
    Init/save/load  are not too surprising. 
    
    fit_grammar will tell the grammar object to learn from the graphs you provide.
    
    fit will call fit_grammar  
        it also tries to train a SVM with all the input graphs that 
        can decide how much a given graph is like the ones in input.
    
    '''

    def __init__(self,
                 radius_list=[0, 1],
                 thickness_list=[1, 2],
                 grammar=None,
                 core_interface_pair_remove_threshold=2,
                 interface_remove_threshold=2,
                 complexity=3,
                 vectorizer=Vectorizer(complexity=3),
                 estimator=estimator_wrapper.estimator_wrapper()):


        self.complexity = complexity
        self.feasibility_checker = FeasibilityChecker()
        self.postprocessor = processing.PostProcessor()
        self.vectorizer = vectorizer
        # lists of int
        self.radius_list = [int(2 * r) for r in radius_list]
        self.thickness_list = [int(2 * t) for t in thickness_list]
        # scikit  classifier
        self.estimatorobject = estimator
        # grammar object
        self.local_substitutable_graph_grammar = grammar
        # cips hashes will be masked with this, this is unrelated to the vectorizer
        self.hash_bitmask = pow(2, 20) - 1
        # we will save current graph at every intervalth step of sampling and attach to graphinfos[graphs]
        self.sampling_interval = None
        # how many sampling steps are done
        self.n_steps = None
        # current step in sampling proces of a single graph
        self.step = None
        # how often do we try to get a cip from the current graph  in sampling
        self.select_cip_max_tries = None
        # sample path
        self.sample_path = None

        self.local_substitutable_graph_grammar = LocalSubstitutableGraphGrammar(self.radius_list,
                                                                                    self.thickness_list,
                                                                                    complexity=self.complexity,
                                                                                    cip_remove_threshold=core_interface_pair_remove_threshold,
                                                                                    interface_remove_threshold=interface_remove_threshold,
                                                                                    nbit=20)
        
    def save(self, file_name):
        self.local_substitutable_graph_grammar._revert_multicore_transform()
        dill.dump(self.__dict__, open(file_name, "w"), protocol=dill.HIGHEST_PROTOCOL)
        # joblib.dump(self.__dict__, file_name, compress=1)
        logger.debug('Saved model: %s' % file_name)

    def load(self, file_name):
        # self.__dict__ = joblib.load(file_name)
        self.__dict__ = dill.load(open(file_name))
        logger.debug('Loaded model: %s' % file_name)

    def fit(self, graphs,
            core_interface_pair_remove_threshold=2,
            interface_remove_threshold=2,
            nu=.5):
        """
          use input to fit the grammar and fit the estimator
        """
        graphs, graphs_ = itertools.tee(graphs)
        self.estimator = self.estimatorobject.fit(graphs_, vectorizer=self.vectorizer, nu=nu)
        self.local_substitutable_graph_grammar.fit(graphs)




    '''
      ENTRY POINT FOR SAMPLING. THE ACTUAL WORK WILL BE DONE BY _SAMPLE
      
    '''
    def sample(self, graph_iter,
               n_samples=10,
               n_steps=50,
               select_cip_max_tries=20):
        """
            input: graph iterator
            output: yield (sampled_graph,{dictionary of info about sampling process}
        """
        self.sampling_interval= 99999
        if n_samples:
            self.sampling_interval = int(n_steps / n_samples) + 1
        self.n_steps = n_steps
        self.select_cip_max_tries = select_cip_max_tries
   
        # sampling
        for graph in graph_iter:
            sampled_graph = self._sample(graph)
            # yield sampled_graph
            for new_graph in self.return_formatter(sampled_graph):
                yield new_graph

    def return_formatter(self, sample_product):
        # after _sample we need to decide what to yield...
        yield sample_product


    '''
    
    HERE ALL THE SAMPLING HAPPENS EXCEPT THE GRAPH PROPOSITION WHICH IS DESCRIBED BELOW
    
    '''
    def _sample(self, graph):
        '''
            we sample a single graph.

            input: a graph
            output: (sampled_graph,{info dictionary})
        '''
        # prepare variables and graph
        graph = self._sample_init(graph)
        self._score_list = [graph._score]
        self.sample_path = []

        try:
            for self.step in xrange(self.n_steps):
                # check similarity - stop condition..
                self._stop_condition(graph)
                # get a proposal for a new graph
                # keep it if we like it
                candidate_graph = self._propose(graph)
                if self._accept(graph, candidate_graph):
                    graph = candidate_graph

                # save score
                # take snapshot
                self._score_list_append(graph)
                self._sample_path_append(graph)

        except Exception as exc:
            logger.debug(exc)
            logger.debug(traceback.format_exc(10))

        self._score_list += [self._score_list[-1]] * (self.n_steps + 1 - len(self._score_list))
        # we put the result in the sample_path
        # and we return a nice graph as well as a dictionary of additional information
        self._sample_path_append(graph)
        sampled_graph = self.vectorizer._revert_edge_to_vertex_transform(graph)
        sampled_graph.graph['sampling_info'] = {'graphs_history': self.sample_path, 'score_history': self._score_list}
        return sampled_graph

    def _score_list_append(self, graph):
        self._score_list.append(graph._score)

    def _sample_path_append(self, graph):
        if self.step % self.sampling_interval == 0:
            graph.graph['score'] = graph._score
            self.sample_path.append(self.vectorizer._revert_edge_to_vertex_transform(graph))

    def _sample_init(self, graph):
        '''
        we prepare the sampling process
        '''
        graph = self.vectorizer._edge_to_vertex_transform(graph)
        self._score(graph)
        return graph

    def _stop_condition(self, graph):
        pass
        
    def _score(self, graph):
        """
        :param graph: a graph
        :return: score of graph
        we also set graph.score_nonlog and graph.score
        """
        if '_score' not in graph.__dict__:
            transformed_graph = self.vectorizer.transform_single(nx.Graph(graph))
            # slow so dont do it..
            # graph.score_nonlog = self.estimator.base_estimator.decision_function(transformed_graph)[0]
            graph._score = self.estimator.predict_proba(transformed_graph)[0,1]
        return graph._score

    def _accept(self, graph_old, graph_new):
        '''
            we took the old graph to generate a new graph by conducting a replacement step.
            now we want to know if this new graph is good enough to take the old ones place.
            in this implementation we use the score of the graph to judge the new graph
        '''

        # first calculate the score ratio between old and new graph.
        score_graph_old = self._score(graph_old)
        score_graph_new = self._score(graph_new)
        score_ratio = score_graph_new / score_graph_old
        # if the new graph scores higher, the ratio is > 1 and we accept
        if score_ratio > 1.0:
            return True
        return score_ratio > random.random()




    '''
        FIRST WE PICK A CIP FROM THE ORIGINAL GRPAH (SEE BELOW)
        THEN WE DECIDE ON A CIP TO REPLACE IT WITH
    '''



    def _propose(self, graph):
        '''
         we wrap the propose single cip, so it may be overwritten some day
        '''
        graph = self._propose_graph(graph)
        if graph is not None:
            return graph
        raise Exception("propose failed.")

    def _propose_graph(self, graph):
        """
        we choose ONE core in the graph and return a valid grpah with a changed core
        note that when we chose the core, we made sure that there would be possible replacements..
        """
        # finding a legit candidate..
        original_cip = self.select_original_cip(graph)

        # see which substitution to make
        candidate_cips = self._select_cips(original_cip)
        for candidate_cip in candidate_cips:
            # substitute and return
            graph_new = core_substitution(graph, original_cip.graph, candidate_cip.graph)
            if self.feasibility_checker.check(graph_new):
                graph_clean(graph_new)
                return self.postprocessor.postprocess(graph_new)


    def _select_cips(self, cip):
        """
        :param cip: the cip we selected from the graph
        :yields: cips found in the grammar that can replace the input cip

        log to debug on fail
        """
        core_hashes = self._get_valid_core_hashes(cip)
        for core_hash in core_hashes:
                yield self.local_substitutable_graph_grammar.grammar[cip.interface_hash][core_hash]


    def _get_valid_core_hashes(self, cip):
        '''
        :param cip: the chip to be replaced
        :return: list of core_hashes of acceptable replacement cips
        '''
        result_list = list(self.local_substitutable_graph_grammar.grammar[cip.interface_hash].keys())
        random.shuffle(result_list)
        return result_list

    '''
        PICK A CIP FROM THE ORIGINAL GRPAH
    '''

    def select_original_cip(self, graph):
        """
        selects a cip from the original graph.
        (we try maxtries times to make sure we get something nice)

        - original_cip_extraction  takes care of extracting a cip
        - accept_original_cip makes sure that the cip we got is indeed in the grammar
        """
        for x in xrange(self.select_cip_max_tries):

            #  get a cip
            cip = self._original_cip_extraction(graph)
            if not cip:
                continue
            cip = cip[0]
            
            # return if the cip is good.
            if self._accept_original_cip(cip):
                return cip

        raise Exception('select_cip_for_substitution failed')


    def  _original_cip_extraction(self,graph):
        '''
        selects the next candidate.
        '''
        # choose random node
        node = random.choice(graph.nodes())
        if 'edge' in graph.node[node]:
            node = random.choice(graph.neighbors(node))
        # random radius and thickness
        radius = random.choice(self.local_substitutable_graph_grammar.radius_list)
        thickness = random.choice(self.local_substitutable_graph_grammar.thickness_list)
        return extract_core_and_interface(node, graph, [radius], [thickness], vectorizer=self.vectorizer)


    def _accept_original_cip(self, cip):
        '''
        :param cip: the cip we need to judge
        :return: good or nogood (bool)
        '''
        # if the cip is in the grammar we are ok.
        if cip.interface_hash in self.local_substitutable_graph_grammar.grammar:
            return True
        return False

