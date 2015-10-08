import itertools
import random
import estimator
#from graphtools import GraphWrapper #extract_core_and_interface, core_substitution, graph_clean, mark_median
import feasibility
from localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar
from multiprocessing import Pool
import dill
import traceback
from eden import grouper
from eden.graph import Vectorizer
from eden.util import serialize_dict
import logging
from utils import draw
import processing

logger = logging.getLogger(__name__)





class GraphLearnSampler(object):

    def __init__(self,
                 nbit=20,

                 vectorizer=Vectorizer(complexity=3),
                 random_state=None,

                 estimator=estimator.Wrapper(),
                 preprocessor=processing.PreProcessor(),
                 postprocessor=processing.PostProcessor(),
                 feasibility_checker = feasibility.FeasibilityChecker(),

                 radius_list=[0, 1],
                 thickness_list=[1, 2],
                 node_entity_check=lambda x, y: True,

                 grammar=None,
                 min_cip_count=2,
                 min_interface_count=2):
        """

        :param nbit: the cip-hashes ( core and interface ) will be this many bit long
        :param complexity: is currently ignored since its an argument of the vectorizer
        :param vectorizer: a eden.graph.vectorizer used to turn graphs into vectors. also provides utils
        :param estimator: is trained on the inout graphs. see implementation


        # gramar+sampling options
        :param radius_list: the cores of a root will have these radii
        :param thickness_list: the cores will extend this much
        :param node_entity_check: decides if a cip is used at all. see example implementation
                check is performed during cip extraction.
                notice difference to the accept cip :)


        # grammar options:
        :param grammar: a grammar -> see implementation of localsubstututablegrammar
        :param min_cip_count:  during the grammar training we need to see a cip this many
            times before adding it to the grammar
        :param min_interface_count: if we dont find this many different cores for this interface,\
        it gets removed.

        :return:
        """


        self.preprocessor=preprocessor
        self.feasibility_checker = feasibility_checker


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
        self.max_core_size_diff = None
        # a similaritythreshold at which to stop sampling.  a value <= 0 will render this useless
        self.similarity = None
        # we will save current graph at every interval step of sampling and attach to graphinfos[graphs]
        self.sampling_interval = None
        # how many sampling steps are done
        self.n_steps = None
        # number of jobs created by multiprocessing  -1 to let python guess how many cores you have
        self.n_jobs = None
        # currently stores information on why the sampling was stopped before n_steps ;
        # will be attached to the graphinfo returned by _sample()
        self._sample_notes = None
        # factor for simulated annealing, 0 means off
        # 1 is pretty strong. 0.6 seems ok
        self.improving_threshold = None
        # current step in sampling process of a single graph
        self.step = None
        self.node_entity_check = node_entity_check

        # how often do we try to get a cip from the current graph  in sampling
        self.select_cip_max_tries = None

        # sample path
        self.sample_path = None

        # sample this many before sampling interval starts
        self.burnin = None

        # is the core chosen by frequency?  (bool)
        self.probabilistic_core_choice = None

        if not grammar:
            self.lsgg = \
                LocalSubstitutableGraphGrammar(self.radius_list,
                                               self.thickness_list,
                                               vectorizer=self.vectorizer,
                                               min_cip_count=min_cip_count,
                                               min_interface_count=min_interface_count,
                                               nbit=self.nbit,
                                               node_entity_check=self.node_entity_check)
        else:
            self.lsgg = grammar

        # will be set before fitting and before sampling
        self.random_state = random_state

        # TODO THE REST OF THE VARS HERE>> THERE ARE QUITE A FEW ONES

    def save(self, file_name):
        self.lsgg._revert_multicore_transform()
        dill.dump(self.__dict__, open(file_name, "w"), protocol=dill.HIGHEST_PROTOCOL)
        logger.debug('Saved model: %s' % file_name)

    def load(self, file_name):
        self.__dict__ = dill.load(open(file_name))
        logger.debug('Loaded model: %s' % file_name)

    def grammar(self):
        return self.lsgg


    def fit(self, input, n_jobs=-1, nu=.5, batch_size=10):
        """
          use input to fit the grammar and fit the estimator
        """

        graphmanagers = self.preprocessor.fit_transform(input, self.vectorizer)

        self.postprocessor.fit(self.preprocessor)
        if self.estimatorobject.status != 'trained':
            self.estimatorobject.fit(graphmanagers,
                                                      vectorizer=self.vectorizer,
                                                      nu=nu,
                                                      n_jobs=n_jobs,
                                                      random_state=self.random_state)
        self.lsgg.fit(graphmanagers, n_jobs, batch_size=batch_size)
        return self


    def sample(self, graph_iter,

               probabilistic_core_choice=True,
               score_core_choice=False,
               max_core_size_diff=-1,

               similarity=-1,
               n_samples=None,
               proposal_probability=False,
               batch_size=10,
               n_jobs=0,

               target_orig_cip=False,
               n_steps=50,
               quick_skip_orig_cip=False,
               improving_threshold=-1,
               improving_linear_start=0,
               accept_static_penalty=0.0,
               select_cip_max_tries=20,
               burnin=0,


               generator_mode=False,
               omit_seed=True,
               keep_duplicates=False):
        """

        :param graph_iter:  seed graphs

        # strategy for choosing a cip inside the grammar
        :param probabilistic_core_choice:  the more of a cip i have seen, the more likely ill use it
        :param score_core_choice:   ill use the estimator to determine how likely every cip is.
        (higher score => higher proba)
        :param max_core_size_diff: ill prefer cips that wont change the size of the graph

        # sampling
        :param n_samples:  how many graphs will i print at the end
        :param batch_size:  how many seeds are going to get batched ( small: overhead, large: needs ram )
        :param n_jobs: ill start this many threads
        :param n_steps: how many samplesteps are conducted
        :param burnin: do this many steps before collecting samples

        :proposal_probability: True to calculate new and old options
        # sampling strategy
        :param target_orig_cip:  we will use the estimator to determine weak regions in the graph that need
        improvement
        :param quick_skip_orig_cip: dont try all the hits in the grammar, if the first fails
        (due to feasibility) we give up
        :param improving_threshold: fraction after which we only accept graphs that improve the score.
        :param improving_linear_start: fraction after which we start linearly penalising bad graphs until we reach impth
        :param accept_static_penalty: so there is a chance to accept a lower scoring grpah,  here we penalize
        :param select_cip_max_tries: how often do we try to get an original cip before declaring the seed done
        :param similarity:  provides the option to kill the sampling  eg if distance to seed is too large.

        # output options
        :param generator_mode: yield every graph on its own   or  put all the n_samples as an argument in the
        last graph
        :param omit_seed: dont record the seed graph
        :param keep_duplicates: duplicates are not recorded


        :return:  yield graphs

        """
        self.proposal_probability = proposal_probability

        if self.proposal_probability:
            raise NotImplementedError("deactivated because its implementation was too ugly")
            #print 'WARNING you set estimate backflow. the implementation is a little sketchy ' \
            #      'so dont try this with weired graphs. '

        self.similarity = similarity



        if probabilistic_core_choice + score_core_choice + max_core_size_diff == -1 > 1:
            raise Exception('choose max one cip choice strategy')

        if n_samples:
            self.sampling_interval = int((n_steps - burnin) / (n_samples + omit_seed - 1)) + 1
        else:
            self.sampling_interval = 9999

        self.n_steps = n_steps
        self.quick_skip_orig_cip = quick_skip_orig_cip
        self.n_jobs = n_jobs
        self.target_orig_cip = target_orig_cip

        # the user doesnt know about edge nodes.. so this needs to be done
        max_core_size_diff = max_core_size_diff * 2
        self.max_core_size_diff = max_core_size_diff

        #  calculating the actual steps for improving :)
        self.improving_threshold = improving_threshold
        if improving_threshold > 0:
            self.improving_threshold = int(self.improving_threshold * self.n_steps)
        self.improving_linear_start = improving_linear_start
        if improving_linear_start > 0:
            self.improving_linear_start = int(improving_linear_start * n_steps)
        self.improving_penalty_per_step = (1 - accept_static_penalty) / float(self.improving_threshold - self.improving_linear_start)

        self.accept_static_penalty = accept_static_penalty
        self.select_cip_max_tries = select_cip_max_tries
        self.burnin = burnin
        self.omit_seed = omit_seed
        self.batch_size = batch_size
        self.probabilistic_core_choice = probabilistic_core_choice
        self.score_core_choice = score_core_choice

        self.generator_mode = generator_mode
        self.keep_duplicates = keep_duplicates
        # adapt grammar to task:
        self.lsgg.preprocessing(n_jobs,

                                max_core_size_diff,
                                probabilistic_core_choice,
                                )

        if score_core_choice:
            self.score_core_choice_dict = {}
            for interface in self.lsgg.productions:
                for core in self.lsgg.productions[interface]:
                    gr = self.lsgg.productions[interface][core].graph.copy()
                    transformed_graph = self.vectorizer.transform_single(gr)
                    score = self.estimatorobject.cal_estimator.predict_proba(transformed_graph)[0, 1]
                    self.score_core_choice_dict[core] = score


        logger.debug(serialize_dict(self.__dict__))

        if self.random_state is not None:
            random.seed(self.random_state)
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
            # for pair in graphlearn_utils.multiprocess(graph_iter,\
            #                                           _sample_multi,self,n_jobs=n_jobs,batch_size=batch_size):
            #    yield pair

    def return_formatter(self, sample_product):
        # after _sample we need to decide what to yield...
        if sample_product is not None:
            if self.generator_mode:
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
        try:
            graph_manager = self._sample_init(graph)
        except Exception as exc:
            logger.debug(exc)
            return None

        self._score_list = [graph_manager._score]
        self.sample_path = []
        accept_counter = 0

        try:
            for self.step in xrange(self.n_steps):
                self._sample_path_append(graph_manager)

                # check stop condition..
                self._stop_condition(graph_manager)

                # get a proposal for a new graph
                # keep it if we like it
                candidate_graph_manager = self._propose(graph_manager)
                if self._accept(graph_manager, candidate_graph_manager):
                    accept_counter += 1
                    graph_manager = candidate_graph_manager

                # save score
                self._score_list_append(graph_manager)

        except Exception as exc:
            logger.debug(exc)
            logger.debug(traceback.format_exc(10))
            self._sample_notes += "\n" + str(exc)
            self._sample_notes += '\nstopped at step %d' % self.step

        self._score_list += [self._score_list[-1]] * (self.n_steps + 1 - len(self._score_list))
        # we put the result in the sample_path
        # and we return a nice graph as well as a dictionary of additional information
        self._sample_path_append(graph_manager, force=True)
        sampled_graph = graph_manager.out()

        sampled_graph.graph['sampling_info'] = {'graphs_history': self.sample_path,
                                                'score_history': self._score_list,
                                                'accept_count': accept_counter,
                                                'notes': self._sample_notes}
        return sampled_graph

    def _score_list_append(self, graphman):
        self._score_list.append(graphman._score)

    def _sample_path_append(self, graphmanager, force=False):

        step0 = (self.step == 0 and self.omit_seed is False)
        normal = self.step % self.sampling_interval == 0 and self.step != 0 and self.step > self.burnin

        # conditions meet?
        if normal or step0 or force:
            # do we want to omit duplicates?
            if not self.keep_duplicates:
                # have we seen this before?
                if graphmanager._score in self._sample_path_score_set:
                    # if so return
                    return
                # else add so seen set
                else:
                    self._sample_path_score_set.add(graphmanager._score)

            # append :) .. rescuing score
            #graph.graph['score'] = graph._score # is never used?
            self.sample_path.append(graphmanager.out())





    def _sample_init(self, graph):
        '''
        we prepare the sampling process

        - first we expand its edges to nodes, so eden will be able wo work its magic on it
        - then we calculate a score for the graph, to see how much we like it
        - we setup the similarity checker stop condition
        - possibly we are in a multiprocessing process, and this class instance hasnt been used before,
          in this case we need to rebuild the postprocessing function .
        '''

        graphman=self.preprocessor.transform([graph])[0]

        graph = graphman.base_graph()
        if self.max_core_size_diff > -1:
            self.seed_size = len(graph)



        self._score(graphman)
        self._sample_notes = ''
        self._sample_path_score_set = set()


        #print 'sample init:',graphman
        #draw.graphlearn_draw(graphman.graph())

        return graphman

    def _stop_condition(self, graphmanager):
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
        graph=graphmanager.base_graph()
        if self.similarity > 0:
            if self.step == 0:
                self.vectorizer._reference_vec = self.vectorizer._convert_dict_to_sparse_matrix(
                    self.vectorizer._transform(0, graph.copy()))
            else:
                similarity = self.vectorizer._similarity(graph, [1])
                if similarity < self.similarity:
                    raise Exception('similarity stop condition reached')

    def _score(self, graphmanager):
        """
        :param graphmanager: a graph
        :return: score of graph
        we also set graph.score_nonlog and graph.score
        """
        if '_score' not in graphmanager.__dict__:
            graphmanager._score= self.estimatorobject.score(graphmanager)
        return graphmanager._score

    def _accept(self, graphman_old, graphman_new):
        '''
            we took the old graph to generate a new graph by conducting a replacement step.
            now we want to know if this new graph is good enough to take the old ones place.
            in this implementation we use the score of the graph to judge the new graph
        '''

        accept_decision = False

        # first calculate the score ratio between old and new graph.
        score_graph_old = self._score(graphman_old)
        score_graph_new = self._score(graphman_new)
        score_ratio = score_graph_new / score_graph_old

        if self.proposal_probability:
            score_ratio *= self.proposal_probability_value

        # if the new graph scores higher, the ratio is > 1 and we accept
        if score_ratio > 1.0:
            accept_decision = True
        else:
            # we now know that the new graph is worse than the old one, but we believe in second chances :)
            # the score_ratio is the probability of being accepted, (see next comment block)
            # there are 2 ways of messing with the score_ratio:
            # 1. the annealing factor will increase the penalty as the sampling progresses
            #       (values of 1 +- .5 are interesting here)
            # 2. a static penalty applies a penalty that is always the same.
            #       (-1 ~ always accept ; +1 ~  never accept)

            if self.improving_threshold > 0 and self.step > self.improving_linear_start:
                penalty = ((self.step - self.improving_linear_start) * float(self.improving_penalty_per_step))
                score_ratio = score_ratio - penalty

            elif self.improving_threshold == 0:
                return False

            score_ratio = score_ratio - self.accept_static_penalty

            # score_ratio is smaller than 1. random.random generates a float between 0 and 1
            # the smaller the score_ratio the smaller the chance of getting accepted.
            accept_decision = (score_ratio > random.random())
        return accept_decision

    def _propose(self, graphman):
        """
         we wrap the propose single cip, so it may be overwritten some day
        """
        graph = self._propose_graph(graphman)
        if graph is not None:
            return graph

        raise Exception("propose failed.. reason is that propose_single_cip failed.")

    def _propose_graph(self, graphman):
        """
        so here is the whole procedure:

        select cip tries MAXTRIES times to find a cip from graph.
        on the way it will yield all the possible original_cips it finds.

        on each we do our best to find a hit in the grammar.
        as soon as we found one replacement that works we are good and return.
        """

        for orig_cip_ctr, original_cip in enumerate(self.select_original_cip(graphman)):
            # see which substitution to make
            candidate_cips = self._select_cips(original_cip, graphman)

            for attempt, candidate_cip in enumerate(candidate_cips):
                choices = len(self.lsgg.productions[candidate_cip.interface_hash].keys()) - 1

                # substitute and return
                new_graph = graphman.core_substitution( original_cip.graph, candidate_cip.graph)

                if self.feasibility_checker.check(new_graph):
                    tmp = self.postprocessor.re_transform_single(new_graph)
                    if tmp:
                        tmp.clean()
                        return tmp
                        '''
                        deactivating the proposal probability. we need a better solution for this
                        '''
                        self.calc_proposal_probability(graphman, new_graph, original_cip)
                        logger.debug("_propose_graph: iteration %d ; core %d of %d ; original_cips tried  %d" %
                                     (self.step, attempt, choices, orig_cip_ctr))
                        new_graph.clean() # i clean only here because i need the interface mark for reverse_dir_prob
                        return new_graph


                if self.quick_skip_orig_cip:
                    break

    def calc_proposal_probability(self, graphman, graphman_new, cip):
        '''

        :param graph:  the old graph
        :param graph_new: the new graph
        :param cip: the old cip is enough since we mainly need the ids of the interface
        :return: options(interface,newgraph)+newgraphlength*average /  options(interface,graph)+oldgraphlen*average

        '''

        def ops(gman, cip_graph):
            counter = 0
            interfacesize=0
            for n, d in cip_graph.nodes(data=True):
                if 'edge' not in d and 'interface' in d:
                    cips = gman.rooted_core_interface_pairs(n, radius_list= self.radius_list, thickness_list=self.thickness_list,
                                             hash_bitmask=self.hash_bitmask, filter=self.node_entity_check)
                    for cip in cips:
                        if cip.interface_hash in self.lsgg.productions:
                            counter += len(self.lsgg.productions[cip.interface_hash])
                    interfacesize+=1
            return counter, interfacesize


        if self.proposal_probability:
            old_opts,interfacesize = ops(graphman, cip.graph)
            new_opts,unused = ops(graphman_new, graphman_new.base_graph())
            average_opts=float(old_opts+new_opts)/2
            old_opts=max(1,old_opts)
            new_opts=max(1,new_opts)
            v1 = new_opts + average_opts* ( len(graphman_new.base_graph())-interfacesize)
            v2 = old_opts + average_opts* ( len(graphman.base_graph())-interfacesize)
            value = float(v1)/v2
            self.proposal_probability_value= value
            logger.debug(  'reverse_direction_modifier: %f' % value )

    def _select_cips(self, cip, graphman):
        """
        :param cip: the cip we selected from the graph
        :yields: cips found in the grammar that can replace the input cip

        log to debug on fail
        """
        if not cip:
            raise Exception('select randomized cips from grammar got bad cip')

        # get core hashes
        core_hashes = self.lsgg.productions[cip.interface_hash].keys()
        if cip.core_hash in core_hashes:
            core_hashes.remove(cip.core_hash)

        # get values and yield accordingly
        values = self._core_values(cip, core_hashes, graphman.base_graph())

        for core_hash in self.probabilistic_choice(values, core_hashes):
            # print values,'choose:', values[core_hashes.index(core_hash)]
            yield self.lsgg.productions[cip.interface_hash][core_hash]

    def _core_values(self, cip, core_hashes, graph):
        core_weights = []

        if self.probabilistic_core_choice:
            for core_hash in core_hashes:
                core_weights.append(self.lsgg.frequency[cip.interface_hash][core_hash])

        elif self.score_core_choice:
            for core_hash in core_hashes:
                core_weights.append(self.score_core_choice_dict[core_hash])

        elif self.max_core_size_diff > -1:
            unit = 100 / float(self.max_core_size_diff + 1)
            goal_size = self.seed_size
            current_size = len(graph)

            for core in core_hashes:
                # print unit, self.lsgg.core_size[core] , cip.core_nodes_count , current_size , goal_size
                predicted_size = self.lsgg.core_size[core] - cip.core_nodes_count + current_size
                value = max(0, 100 - (abs(goal_size - predicted_size) * unit))
                core_weights.append(value)
        else:
            core_weights = [1] * len(core_hashes)

        return core_weights

    def probabilistic_choice(self, values, core_hashes):
        # so you have a list of core_hashes
        # now for every core_hash put a number in a rating list
        # we will choose one according to the probability induced by those numbers
        ratings_sum = sum(values)
        # while there are cores
        while core_hashes and ratings_sum > 0.0:
            # get a random one by frequency
            rand = random.uniform(0.0, ratings_sum)
            if rand == 0.0:
                break
            current = 0.0
            i = -1
            while current < rand:
                current += values[i + 1]
                i += 1
            # yield and delete
            yield core_hashes[i]
            ratings_sum -= values[i]
            del values[i]
            del core_hashes[i]

    def select_original_cip(self, graphman):
        """
        selects a cip from the original graph.
        (we try maxtries times to make sure we get something nice)

        - original_cip_extraction  takes care of extracting a cip
        - accept_original_cip makes sure that the cip we got is indeed in the grammar
        """
        if self.target_orig_cip:
            graphman.mark_median( inp='importance', out='is_good', estimator= self.estimatorobject.estimator )


        #draw.graphlearn(graphman.abstract_graph(), size=10)
        #draw.graphlearn(graphman._abstract_graph, size=10)
        #print graphman

        failcount = 0
        nocip = 0
        for x in range(self.select_cip_max_tries):
            # exteract_core_and_interface will return a list of results,
            # we expect just one so we unpack with [0]
            # in addition the selection might fail because it is not possible
            # to extract at the desired radius/thicknes
            cip = graphman.random_core_interface_pair( radius_list=self.radius_list, thickness_list=self.thickness_list,
                                    hash_bitmask=self.hash_bitmask, filter=self.node_entity_check )
            if not cip:
                nocip += 1
                continue
            cip = cip[0]

            #print cip


            if self._accept_original_cip(cip):
                yield cip
            else:
                failcount += 1


        raise Exception(
            'select_cip_for_substitution failed because no suiting interface was found, \
            extract failed %d times; cip found but unacceptable:%s ' % (failcount + nocip, failcount))





    def _accept_original_cip(self, cip):
        '''
        :param cip: the cip we need to judge
        :return: good or nogood (bool)
        '''

        score_ok = True
        if self.target_orig_cip:
            imp = []
            for n, d in cip.graph.nodes(data=True):
                if 'interface' not in d and 'edge' not in d:
                    imp.append(d['is_good'])

            if (float(sum(imp)) / len(imp)) > 0.9:
                # print imp
                # from utils import draw
                # draw.draw_graph(cip.graph, vertex_label='is_good', secondary_vertex_label='importance')
                score_ok = False

        in_grammar = False
        if len(self.lsgg.productions.get(cip.interface_hash, {})) > 1:
            in_grammar = True

        logger.log(5, 'accept_orig_cip: %r %r' % (score_ok, in_grammar))

        return in_grammar and score_ok


def _sample_multi(what):
    self = dill.loads(what[0])
    graphlist = dill.loads(what[1])
    return [self._sample(g) for g in graphlist]
