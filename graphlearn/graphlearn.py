import networkx as nx
import itertools
import random
import postprocessing
import estimator
from graphtools import extract_core_and_interface, core_substitution, graph_clean
from feasibility import FeasibilityChecker
from localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar
from multiprocessing import Pool
import dill
import traceback
from eden import grouper
from eden.graph import Vectorizer
from eden.util import serialize_dict
import logging
from utils import draw
logger = logging.getLogger(__name__)


class GraphLearnSampler(object):

    def __init__(self,
                 radius_list=[0, 1],
                 thickness_list=[1, 2],
                 nbit=20,
                 complexity=3,
                 vectorizer=Vectorizer(complexity=3),
                 node_entity_check=lambda x, y: True,
                 estimator=estimator.estimator(),
                 grammar=None,
                 core_interface_pair_remove_threshold=2,
                 interface_remove_threshold=2  ):

        self.complexity = complexity
        self.feasibility_checker = FeasibilityChecker()
        self.postprocessor = postprocessing.PostProcessor()

        self.vectorizer = vectorizer

        # lists of int
        self.radius_list = [int(2 * r) for r in radius_list]
        self.thickness_list = [int(2 * t) for t in thickness_list]
        # scikit  classifier
        self.estimatorobject = estimator

        # cips hashes will be masked with this, this is unrelated to the vectorizer
        self.hash_bitmask = pow(2, nbit) - 1
        self.nbit = nbit
        # boolean values to set restrictions on replacement
        self.same_radius = None
        self.same_core_size = None
        # a similaritythreshold at which to stop sampling.  a value <= 0 will render this useless
        self.similarity = None
        # we will save current graph at every intervalth step of sampling and attach to graphinfos[graphs]
        self.sampling_interval = None
        # how many sampling steps are done
        self.n_steps = None
        # number of jobs created by multiprocessing  -1 to let python guess how many cores you have
        self.n_jobs = None
        # currently stores information on why the sampling was stopped before n_steps ; will be attached to the graphinfo
        # returned by _sample()
        self._sample_notes = None
        # factor for simulated annealing, 0 means off
        # 1 is pretty strong. 0.6 seems ok
        self.accept_annealing_factor = None
        # current step in sampling proces of a single graph
        self.step = None
        self.node_entity_check = node_entity_check

        # how often do we try to get a cip from the current graph  in sampling
        self.select_cip_max_tries = None

        # sample path
        self.sample_path = None

        # sample this many before sampling interval starts
        self.burnout = None

        # is the core coosen by frequency?  (bool)
        self.probabilistic_core_choice = None




        if not grammar:
            self.local_substitutable_graph_grammar = LocalSubstitutableGraphGrammar(self.radius_list,
                                                                                    self.thickness_list,
                                                                                    complexity=self.complexity,
                                                                                    core_interface_pair_remove_threshold=core_interface_pair_remove_threshold,
                                                                                    interface_remove_threshold=interface_remove_threshold,
                                                                                    nbit=self.nbit,
                                                                                    node_entity_check=self.node_entity_check)
        else:
            self.local_substitutable_graph_grammar=grammar




        # TODO THE REST OF THE VARS HERE>> THERE ARE QUITE A FEW ONES

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
            n_jobs=-1,
            nu=.5,batch_size=10):
        """
          use input to fit the grammar and fit the estimator
        """
        graphs, graphs_ = itertools.tee(graphs)

        self.estimator = self.estimatorobject.fit(graphs_, vectorizer=self.vectorizer, nu=nu, n_jobs=n_jobs)

        self.local_substitutable_graph_grammar.fit(graphs, n_jobs,batch_size=batch_size)






    def sample(self, graph_iter,
               probabilistic_core_choice=True,
               same_radius=False,
               same_core_size=False,
               similarity=-1,
               n_samples=None,
               batch_size=10,
               n_jobs=0,
               n_steps=50,
               accept_annealing_factor=0,
               accept_static_penalty=0.0,
               select_cip_max_tries=20,
               burnout=0,
               generatormode=False,
               keep_duplicates=False):
        """
            input: graph iterator
            output: yield (sampled_graph,{dictionary of info about sampling process}
        """
        self.same_radius = same_radius
        self.similarity = similarity

        if n_samples:
            self.sampling_interval = int((n_steps - burnout) / n_samples) + 1
        else:
            self.sampling_interval = 9999
        self.n_steps = n_steps
        self.n_jobs = n_jobs
        self.same_core_size = same_core_size
        self.accept_annealing_factor = accept_annealing_factor
        self.accept_static_penalty = accept_static_penalty
        self.select_cip_max_tries = select_cip_max_tries
        self.burnout = burnout
        self.batch_size = batch_size
        self.probabilistic_core_choice = probabilistic_core_choice
        self.generatormode = generatormode
        self.keep_duplicates = keep_duplicates
        # adapt grammar to task:
        self.local_substitutable_graph_grammar.preprocessing(n_jobs, same_radius, same_core_size,
                                                             probabilistic_core_choice)

        logger.debug(serialize_dict(self.__dict__))

        # sampling
        if n_jobs in [0, 1]:
            for graph in graph_iter:
                sampled_graph = self._sample(graph)
                # yield sampled_graph
                for new_graph in self.return_formatter(sampled_graph):
                    yield new_graph
        else:
            if n_jobs > 1:
                pool = Pool(processes=n_jobs)
            else:
                pool = Pool()
            sampled_graphs = pool.imap_unordered(_sample_multi, self._argbuilder(graph_iter))

            for batch in sampled_graphs:
                for sampled_graph in batch:
                    for new_graph in self.return_formatter(sampled_graph):
                        yield new_graph
            pool.close()
            pool.join()
            # for pair in graphlearn_utils.multiprocess(graph_iter,_sample_multi,self,n_jobs=n_jobs,batch_size=batch_size):
            #    yield pair

    def return_formatter(self, sample_product):
        # after _sample we need to decide what to yield...
        if sample_product is not None:
            if self.generatormode:
                # yield all the graphs but jump first because that one is the start graph :)
                for graph in sample_product.graph['sampling_info']['graphs_history'][1:]:
                    yield graph
            else:
                yield sample_product

    def _argbuilder(self, problem_iter):
        # for multiprocessing  divide task into small multiprocessable bites
        s = dill.dumps(self)
        for e in grouper(problem_iter, self.batch_size):
            batch = dill.dumps(e)
            yield (s, batch)

    def _sample(self, graph):
        '''
            we sample a single graph.

            input: a graph
            output: (sampled_graph,{info dictionary})
        '''

        if graph is None:
            return None
        # prepare variables and graph
        graph = self._sample_init(graph)
        self._score_list = [graph._score]
        self.sample_path = []
        accept_counter = 0

        try:
            for self.step in xrange(self.n_steps):
                logger.debug('iteration:%d' % self.step)

                # check similarity - stop condition..
                self._stop_condition(graph)

                # get a proposal for a new graph
                # keep it if we like it
                candidate_graph = self._propose(graph)
                if self._accept(graph, candidate_graph):
                    accept_counter += 1
                    graph = candidate_graph

                # save score
                # take snapshot
                self._score_list_append(graph)
                self._sample_path_append(graph)

        except Exception as exc:
            logger.debug(exc)
            logger.debug(traceback.format_exc(10))
            self._sample_notes += "\n" + str(exc)
            self._sample_notes += '\nstoped at step %d' % self.step

        self._score_list += [self._score_list[-1]] * (self.n_steps + 1 - len(self._score_list))
        # we put the result in the sample_path
        # and we return a nice graph as well as a dictionary of additional information
        self._sample_path_append(graph)
        sampled_graph = self.vectorizer._revert_edge_to_vertex_transform(graph)
        sampled_graph.graph['sampling_info'] = {'graphs_history': self.sample_path, 'score_history': self._score_list,
                                                'accept_count': accept_counter, 'notes': self._sample_notes}
        return sampled_graph

    def _score_list_append(self, graph):
        self._score_list.append(graph._score)

    def _sample_path_append(self, graph):
        # conditions meet?
        #
        if self.step == 0 or (self.step % self.sampling_interval == 0 and self.step > self.burnout):

            # do we want to omit duplicates?
            if not self.keep_duplicates:
                # have we seen this before?
                if graph._score in self._sample_path_score_set:
                    # if so return
                    return
                # else add so seen set
                else:
                    self._sample_path_score_set.add(graph._score)

            # append :) .. rescuing score
            graph.graph['score'] = graph._score
            self.sample_path.append(self.vectorizer._revert_edge_to_vertex_transform(graph))

    def _sample_init(self, graph):
        '''
        we prepare the sampling process

        - first we expand its edges to nodes, so eden will be able wo work its magic on it
        - then we calculate a score for the graph, to see how much we like it
        - we setup the similarity checker stop condition
        - possibly we are in a multiprocessing process, and this class instance hasnt been used before,
          in this case we need to rebuild the postprocessing function .
        '''

        graph = self.vectorizer._edge_to_vertex_transform(graph)
        self._score(graph)
        self._sample_notes = ''
        self._sample_path_score_set = set()

        return graph

    def _stop_condition(self, graph):
        '''
        stop conditioni is per default implemented as a similarity checker, to stop if a certain distance from
        the start graph is reached.

        always check if similarity is relevant.. if so then:

        if current step is zero:

            remember the vectorized object
        else:
            similarity between start graph and current graph is expected to decrease.
            if similarity meassure smaller than the limit, we stop
            because we dont want to drift further
        '''
        if self.similarity > 0:
            if self.step == 0:
                self.vectorizer._reference_vec = self.vectorizer._convert_dict_to_sparse_matrix(
                    self.vectorizer._transform(0, nx.Graph(graph)))
            else:
                similarity = self.vectorizer._similarity(graph, [1])
                if similarity < self.similarity:
                    raise Exception('similarity stop condition reached')

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

        # we now know that the new graph is worse than the old one, but we believe in second chances :)
        # the score_ratio is the probability of being accepted, (see next comment block)
        # there are 2 ways of messing with the score_ratio:
        # 1. the annealing factor will increase the penalty as the sampling progresses
        #       (values of 1 +- .5 are interesting here)
        # 2. a static penalty applies a penalty that is always the same.
        #       (-1 ~ always accept ; +1 ~  never accept)
        score_ratio = score_ratio - (  (float(self.step)/self.n_steps) * self.accept_annealing_factor  )
        score_ratio = score_ratio - self.accept_static_penalty

        # score_ratio is smaller than 1. random.random generates a float between 0 and 1
        # the smaller the score_ratio the smaller the chance of getting accepted.
        return score_ratio > random.random()

    def _propose(self, graph):
        '''
         we wrap the propose single cip, so it may be overwritten some day
        '''
        graph = self._propose_graph(graph)
        if graph is not None:
            return graph

        raise Exception("propose failed.. reason is that propose_single_cip failed.")

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
            else:
                logger.debug('feasibility checker failed')

    def _select_cips(self, cip):
        """
        :param cip: the cip we selected from the graph
        :yields: cips found in the grammar that can replace the input cip

        log to debug on fail
        """
        if not cip:
            raise Exception('select randomized cips from grammar got bad cip')

        core_hashes = self._get_valid_core_hashes(cip)
        logger.debug('Working with %d cores' % len(core_hashes))

        if self.probabilistic_core_choice:
            # get all the frequencies
            frequencies = []
            for core_hash in core_hashes:
                frequencies.append(self.local_substitutable_graph_grammar.frequency[cip.interface_hash][core_hash])

            frequencies_sum = sum(frequencies)

            # while there are cores
            while core_hashes:
                # get a random one by frequency
                rand = random.randint(0, frequencies_sum)
                current = 0.0
                i = -1
                while current < rand:
                    current += frequencies[i + 1]
                    i += 1
                # yield and delete
                yield self.local_substitutable_graph_grammar.grammar[cip.interface_hash][core_hashes[i]]
                frequencies_sum -= frequencies[i]
                del frequencies[i]
                del core_hashes[i]

        else:
            for core_hash in core_hashes:
                yield self.local_substitutable_graph_grammar.grammar[cip.interface_hash][core_hash]

        raise Exception("select_randomized_cips_from_grammar didn't find any acceptable cip in ")

    def _get_valid_core_hashes(self, cip):
        '''
        :param cip: the chip to be replaced
        :return: list of core_hashes of acceptable replacement cips
        '''

        if self.same_radius:
            result_list = list(self.local_substitutable_graph_grammar.radiuslookup[cip.interface_hash][cip.radius])
            # if both are activated..
            if self.same_core_size:
                result_list2 = list(
                    self.local_substitutable_graph_grammar.core_size[cip.interface_hash][cip.core_nodes_count])
                result = []
                for hash in result_list2:
                    if hash in result_list:
                        result.append(hash)
                result_list = result
        elif self.same_core_size:
            result_list = list(
                self.local_substitutable_graph_grammar.core_size[cip.interface_hash][cip.core_nodes_count])
        else:
            result_list = list(self.local_substitutable_graph_grammar.grammar[cip.interface_hash].keys())

        random.shuffle(result_list)
        return result_list







    def select_original_cip(self, graph):
        """
        selects a cip from the original graph.
        (we try maxtries times to make sure we get something nice)

        - original_cip_extraction  takes care of extracting a cip
        - accept_original_cip makes sure that the cip we got is indeed in the grammar
        """

        failcount = 0
        nocip= 0
        for x in xrange(self.select_cip_max_tries):
            # exteract_core_and_interface will return a list of results, we expect just one so we unpack with [0]
            # in addition the selection might fail because it is not possible to extract at the desired radius/thicknes
            #
            cip = self._original_cip_extraction(graph)

            if not cip:
                nocip += 1
                continue
            cip = cip[0]
            #print node,radius,cip.interface_hash

            if self._accept_original_cip(cip):
                return cip
            else:
                failcount += 1

        raise Exception(
                'select_cip_for_substitution failed because no suiting interface was found, extract failed %d times; cip found but unacceptable:%s ' % 
            ( failcount+nocip,failcount))


    def  _original_cip_extraction(self,graph):
        '''
        selects the next candidate.
        '''
        node = random.choice(graph.nodes())
        if 'edge' in graph.node[node]:
            node = random.choice(graph.neighbors(node))
            # random radius and thickness
        radius = random.choice(self.local_substitutable_graph_grammar.radius_list)
        thickness = random.choice(self.local_substitutable_graph_grammar.thickness_list)

        return extract_core_and_interface(node, graph, [radius], [thickness], vectorizer=self.vectorizer,
                                             hash_bitmask=self.hash_bitmask, filter=self.node_entity_check)


    def _accept_original_cip(self, cip):
        '''
        :param cip: the cip we need to judge
        :return: good or nogood (bool)
        '''

        #cips=[cip]
        #gr=draw.cip_to_graph( cips )
        #draw.draw_graph_set_graphlearn(gr )
        # if we have a hit in the grammar
        if cip.interface_hash in self.local_substitutable_graph_grammar.grammar:
            #  if we have the same_radius rule implemented:
            if self.same_radius:
                # we jump if that hit has not the right radius
                if not self.local_substitutable_graph_grammar.radiuslookup[cip.interface_hash][cip.radius]:
                    return False
            if self.same_core_size:
                if cip.core_nodes_count not in self.local_substitutable_graph_grammar.core_size[cip.interface_hash]:
                    return False
            return True
        return False


def _sample_multi(what):
    self = dill.loads(what[0])
    graphlist = dill.loads(what[1])
    return [self._sample(g) for g in graphlist]
