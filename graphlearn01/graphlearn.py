'''
graphsampler, takes graphs in input and generates graphs with similar properties
'''
import random
import time
import estimate
import feasibility
from localsubstitutablegraphgrammar import LocalSubstitutableGraphGrammar
from multiprocessing import Pool
import dill
import traceback
from eden import grouper
from eden.graph import Vectorizer
from eden.util import serialize_dict
from utils import draw
import transform
import decompose

import logging
logger = logging.getLogger(__name__)
import utils.monitor as monitor
import networkx as nx
import copy
from cip_select import select_original_cip, _select_cips

import utils



class Sampler(object):
    def __neg__(self):
        '''
        samplers are connectable via + and - symbols.
        this takes care of negating a sampler.
        Returns
        -------
            negating a self means to revert all the estimator results
        '''

        duplicate = copy.deepcopy(self)
        duplicate._nocopy_negate()
        return duplicate

    def _nocopy_negate(self):
        ''''
        helper for __neg__
        '''
        self.estimatorobject.inverse_prediction = not self.estimatorobject.inverse_prediction
        if 'spawn_list' in self.__dict__:
            for spawn in self.spawn_list:
                spawn._nocopy_negate()

    def __mul__(self, other):
        '''
        samplers are connectable via + and - symbols.
        mul is currently not supported
        Returns
        -------
            error
        '''
        # other musst be int oO
        # ==> also apply to its children
        # if 'multiplier' not in self.__dict__:
        #    self.multiplier=1
        # self.multiplier*=other
        raise Exception('multiplication is basically n_steps ooO')
        # if 'spawn_list' in self.__dict__:
        #    for spawn in self.spawn_list:
        #        spawn.__multi__(other)
        # return self

    def __sub__(self, other):
        '''
        samplers are connectable via + and - symbols.
        this takes care of subtraction.
        Returns
        -------
            a sampler that has child samplers in self.spawn_list
        '''
        duplicate = copy.deepcopy(self)
        return duplicate.__add__(other.__neg__())

    def __add__(self, other):
        '''
        samplers are connectable via + and - symbols.
        this takes care of adding samplers.
        Returns
        -------
            a sampler that has child samplers in self.spawn_list
        '''
        duplicate = copy.deepcopy(self)
        if 'spawn_list' not in duplicate.__dict__:
            duplicate.spawn_list = []
        duplicate.spawn_list.append(other)
        return duplicate

    def set_params(self, **parameters):
        '''
        http://scikit-learn.org/stable/developers/contributing.html

        Parameters
        ----------
        parameters: dict {param, value}

        Returns
        -------
            self
        '''
        self.__dict__.update(parameters)
        self._init_new_params()
        return self

    def __init__(self,

                 vectorizer=Vectorizer(complexity=3 ),
                 random_state=None,
                 estimator=estimate.OneClassEstimator(nu=.5, cv=2, n_jobs=-1),
                 graphtransformer=transform.GraphTransformer(),
                 feasibility_checker=feasibility.FeasibilityChecker(),
                 decomposer=decompose.Decomposer(node_entity_check=lambda x, y: True, nbit=20),
                 grammar=LocalSubstitutableGraphGrammar(radius_list=[0, 1],
                                                        thickness_list=[1, 2],
                                                        min_cip_count=2,
                                                        min_interface_count=2),
                 size_diff_core_filter=-1,
                 core_choice_byfrequency=False,
                 core_choice_byscore=False,
                 core_choice_bytrial=False,
                 core_choice_bytrial_multiplier=1.0,

                 size_constrained_core_choice=-1,
                 similarity=-1,
                 n_samples=None,
                 proposal_probability=False,
                 batch_size=10,
                 n_jobs=0,

                 orig_cip_max_positives=1,
                 orig_cip_min_positives=0,

                 n_steps=50,
                 quick_skip_orig_cip=False,
                 improving_threshold_fraction=-1,
                 improving_linear_start_fraction=0,
                 accept_static_penalty=0.0,
                 accept_min_similarity=0.0,
                 select_cip_max_tries=25,
                 burnin=0,
                 backtrack=0,

                 include_seed=False,
                 keep_duplicates=False,

                 monitor=False,
                 proposed_graphs_per_step = 1
                 ):

        '''
        init for graphlearn




                to emulate MCMC sampling use these options:
        probabilistic_core_choice=False?
        score_core_choice=False
        max_size_diff=-1
        proposal_probability=False
        target_orig_cip=False
        improving_threshold=-1
        improving_linear_start=1
        accept_static_penalty=0.0
        burnin=0
        backtrack=0



        Parameters
        ----------
        nbit : int
            the cip-hashes ( core and interface ) will be this many bit long
        vectorizer : vectorizer
            a eden.graph.vectorizer used to turn graphs into vectors. also provides utils
        random_state : int
            random seed
        estimator : estimator.wrapper
            an estimator trained or untrained oOo

        graphtransformer : graphlearn.processing.preprocessor
        postprocessor : graphlearn.processing.postprocessor
        feasibility_checker : feasibility.FeasibilityChecker()
            can determine if a graph is valid
        radius_list : list
        thickness_list : list
        node_entity_check : lambda x,y: bool
        grammar : localsubstitutablegraphgrammar
            a grammar
        min_cip_count : int
            how often do i need to see a cip to accept it into the grammar
        min_interface_count: int
            how many cips need to be in an interface




        size_diff_core_filter: int
            filters the cores for size before the other core_choices are applied.
            i just hard-filter all the cips whose implantation would result in a graph thats not in +- arg of the seed
        core_choice_byfrequency : bool
            cores are chosen according to their frequency in the grammar...
        core_choice_byscore : bool
            cores are chosen  probabilisticaly according to their score
        size_constrained_core_choice : int
            linear increasing penalty is applied to enforce that the graphs
            stays in the desired size range

        similarity : float
            stop condition for sampling, stop if desired similarity is reached,
            similarity meassure is weired due to high dimensionality of eden vector, be warned
        n_samples : int
            collect this many samples for each seed graph
        proposal_probability : bool
            if you are not dealing with abstract graphs you get this option;
            if you want to comply to Metropolis hastings
        batch_size : int
            this many graphs will be processed by one instance,
            (maybe i should calculate the max effective number and use that)
        n_jobs : int (-1)
            number of processes created used. -1 is cpu count
        orig_cip_max_positives : float , -1
        orig_cip_min_positives : float , -1
            eden will evaluate the graph and assign a score to each node.
            all nodes in the better half are marked as high scoring nodes.
            min/max positives is the ratio of allowed high-scoring nodes
            in the core of the cip picked to be replaced.
            .5-> 50% high scoring nodes in core of cip

        n_steps: int
            sample steps

        quick_skip_orig_cip : bool
            for each cip on the original graph, only try one entry from the grammar.

        improving_threshold : float
            starting from this fraction we only accept a graph if it is better
            note that zero is, as you would expect, only accepting better graphs.
        improving_linear_start : float
            starting from this fraction there is a linearly increasing penalty
            to the score until the improving_threshould value
        accept_static_penalty : float
            decrease probability of accepting a worse graph
        accept_min_similarity : in [0,1]
            acceptance requirement, graphs musst be at least this similar to be accepted..
            zero is ignore this
        select_cip_max_tries : int
            try this many times to get a cip from the original graph before declaring
            the seed dead.
        burnin : int
            ignore this many graphs until n_samples starts collecting
        backtrack : int
            sometimes you  generate a dead-end graph, a graph that is valid but finding a proposal is impossible.
            you can take one step back this many times.
            this is of questionable efficiency currently because we cant detecect
            the exact place where we went wrong.
        include_seed : bool
            dont collect the seed as sample
        keep_duplicates : bool
            metropolice compliance says that we should output duplicates. but otherwise duplicates
            are not interesting.
        monitor : bool
            enabling monitor accessible after  sampling. sampler.monitors will contain all the information

       Returns
        -------
        an initialized sampler

        '''
        self.core_choice_bytrial_multiplier=core_choice_bytrial_multiplier
        self.graph_transformer = graphtransformer
        self.feasibility_checker = feasibility_checker
        self.vectorizer = vectorizer
        # scikit  classifier
        self.estimatorobject = estimator
        self.maxbacktrack = backtrack
        self.monitor = monitor
        self.monitors = []
        self.accept_min_similarity = accept_min_similarity
        self.proposal_probability = proposal_probability
        self.similarity = similarity
        self.probabilistic_core_choice = core_choice_byfrequency
        self.core_choice_bytrial = core_choice_bytrial
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.quick_skip_orig_cip = quick_skip_orig_cip
        self.n_jobs = n_jobs
        self.orig_cip_max_positives = orig_cip_max_positives
        self.orig_cip_min_positives = orig_cip_min_positives
        self.size_diff_core_filter = size_diff_core_filter
        # the user doesnt know about edge nodes.. so this needs to be done
        self.improving_threshold_fraction = improving_threshold_fraction
        self.improving_linear_start_fraction = improving_linear_start_fraction
        self.accept_static_penalty = accept_static_penalty
        self.select_cip_max_tries = select_cip_max_tries
        self.burnin = burnin
        self.include_seed = include_seed
        self.batch_size = batch_size
        self.score_core_choice = core_choice_byscore
        self.keep_duplicates = keep_duplicates
        self.lsgg = grammar
        self.random_state = random_state
        self.decomposer = decomposer
        self.size_constrained_core_choice = size_constrained_core_choice
        self.proposed_graphs_per_step = proposed_graphs_per_step


        # init, since someone might call set_param which might also require a reinit.
        self._init_new_params()

    def _init_new_params(self):

        self.improving_threshold_absolute = self.n_steps
        self.improving_linear_start_absolute = 0

        self.orig_cip_score_tricks = self.orig_cip_max_positives != 1 or self.orig_cip_min_positives != 0

        if self.improving_linear_start_fraction > 0:
            self.improving_linear_start_absolute = int(self.improving_linear_start_fraction * self.n_steps)

        # calculating the actual steps for improving :)
        if self.improving_threshold_fraction > 0:
            self.improving_threshold_absolute = int(self.improving_threshold_fraction * self.n_steps)

        if self.improving_linear_start_absolute == self.improving_threshold_absolute:
            self.improving_penalty_per_step = 1
            # the main thing happens in the else branch, the if branch catches the divByZero
        else:
            self.improving_penalty_per_step = (1 - self.accept_static_penalty) / float(
                self.improving_threshold_absolute - self.improving_linear_start_absolute)

        if self.core_choice_bytrial + self.probabilistic_core_choice + self.score_core_choice + (
                    self.size_constrained_core_choice > -1) == 0:
            self.probabilistic_core_choice = True

        if self.core_choice_bytrial + self.probabilistic_core_choice + self.score_core_choice + (
            self.size_constrained_core_choice > -1) > 1:
            raise Exception('choose only one parameter core_choice')


        if self.n_samples:
            self.sampling_interval = int((self.n_steps - self.burnin) / (self.n_samples + self.include_seed - 1))
        else:
            self.sampling_interval = 9999

        #logger.debug(serialize_dict(self.__dict__))

        if self.random_state is not None:
            random.seed(self.random_state)


    def _init_grammar_prep(self):
        # adapt grammar to task:
        self.lsgg.preprocessing(n_jobs=self.n_jobs,
                                core_size_required=(
                                                   self.size_constrained_core_choice + self.size_diff_core_filter) > -2,
                                probabilistic_core_choice=self.probabilistic_core_choice,
                                score_cores=self.score_core_choice,
                                score_cores_vectorizer=self.vectorizer,
                                score_cores_estimator=self.estimatorobject,
                                bytrial=self.core_choice_bytrial)

    def set_parmas(self, **params):
        '''
        Parameters
        ----------
        params:
        Returns
        -------
        '''
        self.__dict__.update(params)
        self._init_new_params()

    def save(self, file_name):
        self.lsgg._revert_multicore_transform()
        dill.dump(self.__dict__, open(file_name, "w"), protocol=dill.HIGHEST_PROTOCOL)
        logger.debug('Saved model: %s' % file_name)

    def load(self, file_name):
        self.__dict__ = dill.load(open(file_name))
        logger.debug('Loaded model: %s' % file_name)

    def grammar(self):
        return self.lsgg

    def fit_transformer(self, graphs):
        self.graph_transformer.fit(graphs)

    def fit_make_decomposers(self, graphs):
        return [self.decomposer.make_new_decomposer(data)
                for data in self.graph_transformer.transform(graphs)]

    def fit_grammar(self, decomposers, n_jobs=-1, batch_size=10):
        self.lsgg.fit(decomposers, n_jobs=n_jobs, batch_size=batch_size)
        # self._init_grammar_prep() cant do it here cuz esti might not be ready


    def decomps_to_graphs(self,decomposers):
        res = [d.pre_vectorizer_graph() for d in decomposers]
        return res

    def decomps_to_vectors(self,decomposers):
        return self.vectorizer.transform(self.decomps_to_graphs(decomposers))

    def fit_estimator(self, decomposers, negative_decomposers=[], regression_targets=None,**args):
        positive = self.decomps_to_vectors(decomposers)
        if negative_decomposers == [] and regression_targets == None:

            # draw.graphlearn(positive[:5], contract=False)
            # print positive[0].graph
            self.estimatorobject.fit(positive,
                                     random_state=self.random_state)
        elif negative_decomposers == None:
            self.estimatorobject = estimate.Regressor()
            self.estimatorobject.fit(positive, regression_targets,
                                     random_state=self.random_state)
        else:
            # twoclass
            negative = self.decomps_to_vectors(negative_decomposers)#[d.pre_vectorizer_graph() for d in negative_decomposers]
            self.estimatorobject.fit(positive, negative,
                                     #random_state=self.random_state,
                                     **args)

    def fit(self, graphs):
        fit_time=time.time()
        decomposers = [self.decomposer.make_new_decomposer(data)
                       for data in self.graph_transformer.fit_transform(graphs)]
        self.fit_grammar(decomposers)
        self.fit_estimator(decomposers)
        logger.debug('sampler fit time: '+str(time.time()-fit_time) )

    def fit_transform(self, graphs):

        graphs = [g for g in graphs]
        graphs2 = copy.deepcopy(graphs)
        self.fit(graphs)
        for out in self.transform(graphs2):
            yield out

    def transform(self, graph_iter=None):
        '''
        starting the sample process

        graph_iter: iterator over networkx graphs

        Returns
        -------
            lists of graphs
        '''
        self._init_grammar_prep()  # this has to be here since fittin is now all split up and i cant controll the oder in which things happen

        if self.n_jobs in [0, 1]:
            for out in self._single_process(graph_iter):
                yield out
        else:
            for out in self._multi_process(self.n_jobs, graph_iter):
                yield out

    def _multi_process(self, n_jobs, graph_iter):
        if n_jobs > 1:
            pool = Pool(processes=n_jobs)
        else:
            # -1
            pool = Pool()

        sampled_graphs = pool.imap_unordered(_sample_multi, self._make_multi_process_batches(graph_iter))
        #sampled_graphs = map(_sample_multi, self._make_multi_process_batches(graph_iter))

        jobs_done = 0
        for batch in sampled_graphs:
            for graphlist, moni in batch:
                moni = dill.loads(moni)
                # dill.loads(what[0])
                # print type(graph)
                # currently formatter only returns one element and thats 
                #fine, one day this may be changed
                
                for new_graph in self._return_formatter(graphlist, moni):
                    yield new_graph

                # forcing termination once the results are in.
                jobs_done += 1
                # python is already starting jobs while not all are in the queue
                if jobs_done == self.multiprocess_jobcount and self.multiprocess_all_prepared:
                    pool.terminate()

        pool.close()
        pool.join()
        # for pair in graphlearn_utils.multiprocess(graph_iter,\
        #                                           _sample_multi,self,n_jobs=n_jobs,batch_size=batch_size):
        #    yield pair

    def _single_process(self, graph_iter):
        for graph in graph_iter:
            # sampled_graph = self._sample(graph)
            # yield sampled_graph
            graphlist, monitor = self.transform_single(graph)
            monitor = dill.loads(monitor)
            for new_graph in self._return_formatter(graphlist, monitor):
                yield new_graph

    def _return_formatter(self, graphlist, mon):
        '''
        this function is here so the output format can be altered to anything.

        Parameters
        ----------
        graphlist: list of graphs
        mon: monitor object assiciated with this sampling run

        Returns
        -------
            output of a run
        '''

        self.monitors.append(mon)
        yield graphlist

    def _make_multi_process_batches(self, problem_iter):
        '''
        we do two things here:
        -break tasks into batches to be multiprocessed.
        -multiprocess sometimes does not terminate properly so we observe how many tasks go in and terminate
        once that number of outs is reached.

        Parameters
        ----------
        problem_iter: problems to put into the multiprocess queue

        Returns
        -------
            yields a batchsize sized problem chunks
        '''
        try:
            s = dill.dumps(self, byref=False)
        except Exception as exc:
            print exc
            print "dill dump failed in graphlearn.py (dill dies silently sometimes)"
        self.multiprocess_jobcount = 0
        self.multiprocess_all_prepared = False

        for e in grouper(problem_iter, self.batch_size):
            # cant just take batch size here because output of nons will be suppressed
            problems = [1 for problem in e if problem != None]
            self.multiprocess_jobcount += sum(problems)
            batch = dill.dumps(e)
            yield (s, batch)
        self.multiprocess_all_prepared = True

    def _samplelog(self, msg, level=10):
        '''
        use this for logging.
        logs are logged and written to monitor.

        Parameters
        ----------
        msg: string
        level: importance of message

        Returns
        -------
            nothing
        '''
        # debug messages in _sample will use this,
        # we will also log to monitor.
        logger.log(level, msg)
        self.monitorobject.info('debug', 'debuglevel:%d %s' % (level, msg))

    def transform_single(self, graph):
        '''
            we sample a single graph.
            input: a graph
            output: ??
        '''

        logger.log(5,"processing a grpah:\n\n\n\n")
        if graph is None:
            return None
        # prepare variables and graph
        try:
            graph_decomposer = self._sample_init(graph)
        except Exception as exc:
            logger.warning(exc)
            logger.warning(traceback.format_exc(10))
            return None

        self._score_list = [graph_decomposer._score]
        self.sample_path = []
        accept_counter = 0
        self.step = 0
        self.monitorobject.tick(graph_decomposer, self.step)
        try:
            while self.step < self.n_steps:
                self._sample_path_append(graph_decomposer)
                # check stop condition..
                self._stop_condition(graph_decomposer)

                # get a proposal for a new graph
                # keep it if we like it
                candidate_graph_decomposer = self._choose_proposal(graph_decomposer)
                if self._accept(graph_decomposer, candidate_graph_decomposer):
                    accept_counter += 1
                    graph_decomposer = candidate_graph_decomposer

                # save score
                self._score_list_append(graph_decomposer)
                self.monitorobject.tick(candidate_graph_decomposer, self.step + 1)
                self.step += 1

        except Exception as exc:
            #self._samplelog(exc)
            self._samplelog(traceback.format_exc(10),level=5)
            self._samplelog('transform_single stopped at %d out of %d n_steps' % (self.step, self.n_steps))
            self._sample_notes += '\nstopped at step %d' % self.step

        self._score_list += [self._score_list[-1]] * (self.n_steps + 1 - len(self._score_list))
        # we put the result in the sample_path
        # and we return a nice graph as well as a dictionary of additional information
        self._sample_path_append(graph_decomposer, force=True)

        sampling_info = {'score_history': self._score_list,
                         'accept_count': accept_counter,
                         'notes': self._sample_notes}
        self.monitorobject.sampling_info = sampling_info
        return self.sample_path, dill.dumps(self.monitorobject)


    def _choose_proposal(self,graph_decomposer):
       props= [self._propose(graph_decomposer) 
                        for x in range( self.proposed_graphs_per_step)]

       #draw.graphlearn(map(lambda x:x._abstract_graph, props), contract=False, vertex_label='id', secondary_vertex_label='contracted')
       #draw.graphlearn(map(lambda x:x._base_graph, props), contract=False, vertex_label='id', secondary_vertex_label='contracted')

       scores = map(self._score, props)
        
       return props[scores.index( max(scores)  )]




    def _score_list_append(self, decomposer):
        '''
        adds score of graph-decomposer to the score_list that will be accessible
        through the monitor at the end.

        Parameters
        ----------
        decomposer: a graph decomposer

        Returns
        -------
            adds score of graphman to the score_list that will be written to the monitor.
        '''
        self._score_list.append(decomposer._score)

    def _sample_path_append(self, decomposer, force=False):
        '''
        decide if we record a speciffic graph.
        this is mostly dependant on the current step.
        sample path is part of the output.

        Parameters
        ----------
        decomposer: a decomposer

        force: bool
            if true force the appending

        Returns
        -------
            nothing
        '''
        # self.include_seed was checking for wfalse... wtf...
        step0 = (self.step == 0 and self.include_seed is True)

        normal = self.step % self.sampling_interval == 0 and self.step != 0 and self.step > self.burnin

        # conditions meet?
        if normal or step0 or force:
            # do we want to omit duplicates?
            if not self.keep_duplicates:
                # have we seen this before?
                if decomposer._score in self._sample_path_score_set:
                    # if so return
                    return
                # else add so seen set
                else:
                    self._sample_path_score_set.add(decomposer._score)

            # append :) .. rescuing score
            # graph.graph['score'] = graph._score # is never used?
            self.sample_path.append(decomposer.out())

    def _sample_init(self, graph):
        '''
        we prepare the sampling process

        - first we expand its edges to nodes, so eden will be able wo work its magic on it
        - then we calculate a score for the graph, to see how much we like it
        - we setup the similarity checker stop condition
        - possibly we are in a multiprocessing process, and this class instance hasnt been used before,
          in this case we need to rebuild the postprocessing function .
        '''
        self._sample_init_init_monitor()
        self.backtrack = self.maxbacktrack
        self.last_decomposer = None
        decomposer = self.decomposer.make_new_decomposer(self.graph_transformer.transform([graph])[0])

        graph = decomposer.base_graph()
        if self.size_constrained_core_choice > -1 or self.size_diff_core_filter > -1:
            self.seed_size = len(graph)
        self._score(decomposer)
        self._sample_notes = ''
        self._sample_path_score_set = set()
        if self.include_seed == False:  # make sure that seed never appears,, may happen if there is nothing happening
            self._sample_path_score_set.add(decomposer._score)
        # print 'sample init:',graphman
        # draw.graphlearn_draw(graphman.graph())
        return decomposer

    def _sample_init_init_monitor(self):
        '''
        reinitializes the monitor.
        normally this is called after each sample-run.

        Returns
        -------
            nothing
        '''
        self.monitorobject = monitor.Monitor(self.monitor)

    def _stop_condition(self, decomposer):
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
        graph = decomposer.base_graph()
        if self.similarity > 0:
            if self.step == 0:
                self.vectorizer._reference_vec = self.vectorizer._convert_dict_to_sparse_matrix(
                    self.vectorizer._transform(0, graph.copy()))
            else:
                similarity = self.vectorizer._similarity(graph, [1])
                if similarity < self.similarity:
                    raise Exception('similarity stop condition reached')

    def _score(self, decomposer):
        """
        will determine the score of a graph.
        scores will be cached

        Parameters
        ----------
        decomposer: a graph-decomposer

        Returns
        -------
        score of graph
        """

        if 'vectorized_graph' not in decomposer.__dict__:
            #decomposer.vectorized_graph = self.vectorizer.transform([decomposer.pre_vectorizer_graph()])
            decomposer.vectorized_graph = self.vectorizer.transform([decomposer.pre_vectorizer_graph()])

        if '_score' not in decomposer.__dict__:
            decomposer._score = self.estimatorobject.predict(decomposer.vectorized_graph)
            self.monitorobject.info('score', decomposer._score)

        return decomposer._score

    def _accept(self, decomposer_old, decomposer_new):
        '''
            we took the old graph to generate a new graph by conducting a replacement step.
            now we want to know if this new graph is good enough to take the old ones place.
            in this implementation we use the score of the graph to judge the new graph
        '''

        accept_decision = False

        # first calculate the score ratio between old and new graph.
        score_graph_old = self._score(decomposer_old)
        score_graph_new = self._score(decomposer_new)
        if self.accept_min_similarity:
            res = decomposer_new.transformed_vector.dot(decomposer_old.transformed_vector.T).todense()
            prediction = res[0, 0]
            if prediction < self.accept_min_similarity:
                return False

        score_ratio = score_graph_new / score_graph_old

        # this is to make sure that the forward and backward prop in the markovchain are similar
        if self.proposal_probability:
            score_ratio *= self.proposal_probability_value
        # updating the trial core weighting rule
        if self.core_choice_bytrial:
            score_diff =( score_graph_new - score_graph_old) / score_graph_old
            self.lsgg.bytrial_update(self.last_cip, score_diff* self.core_choice_bytrial_multiplier)



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

            if self.improving_threshold_absolute > 0 and self.step > self.improving_linear_start_absolute:
                penalty = ((self.step - self.improving_linear_start_absolute) * float(self.improving_penalty_per_step))
                score_ratio = score_ratio - penalty

            elif self.improving_threshold_absolute == 0:
                return False

            score_ratio = score_ratio - self.accept_static_penalty
            # score_ratio is smaller than 1. random.random generates a float between 0 and 1
            # the smaller the score_ratio the smaller the chance of getting accepted.
            accept_decision = (score_ratio > random.random())

        self.monitorobject.info('accepted:', accept_decision)
        return accept_decision

    def _propose(self, decomposer):
        '''
        proposes an altered graph,
        _propose_graph does the actual proposition.
        here we just take care of the possibility to backtrack once we reach a dead end.

        Parameters
        ----------
        decomposer: graph that will be altered a little.

        Returns
        -------
            proposed decomposer
        '''
        if self.maxbacktrack > 0:
            self.backtrack_decomposer = self.last_decomposer
            self.last_decomposer = decomposer

        proposed_decomposer = self._propose_graph(decomposer)

        if self.backtrack > 0 and not proposed_decomposer:
            # print 'backtracking'
            # draw.graphlearn([graphman.base_graph(),self.backtrack_graphman.base_graph()])
            self.backtrack -= 1
            self.step -= 1
            self.monitorobject.info('backtrack to (score)', self.backtrack_decomposer._score)
            proposed_decomposer = self._propose_graph(self.backtrack_decomposer)

        if proposed_decomposer:
            return proposed_decomposer

        raise Exception("propose failed.. _propose_graph returned nothing")

    def _propose_graph(self, decomposer):
        """
        so here is the whole procedure:

        select cip tries MAXTRIES times to find a cip from graph.
        on the way it will yield all the possible original_cips it finds.

        on each we do our best to find a hit in the grammar.
        as soon as we found one replacement that works we are good and return.
        """


        # ORIG CIPS
        for orig_cip_ctr, original_cip in enumerate(select_original_cip(decomposer, self)):

            # CANDIDATES
            candidate_cips = _select_cips(original_cip, decomposer, self)
            attempt='no candidates for this orig cip' # -> set for the logger.
            for attempt, candidate_cip in enumerate(candidate_cips):



                # SUBSTITUTE
                choices = len(self.lsgg.productions[candidate_cip.interface_hash].keys()) - 1
                self.monitorobject.info('substitution', "root: %d , newcip: %d / %d" %
                                        (original_cip.distance_dict[0][0], candidate_cip.interface_hash,
                                         candidate_cip.core_hash))
                new_graph = decomposer.core_substitution(original_cip.graph, candidate_cip.graph)
                self.last_cip=candidate_cip # required for the bytrial flag (adjusts core pick rate by graphscoredelta)


                # FEASIBILITY
                if self.feasibility_checker.check(new_graph):
                    new_decomposer = self.decomposer.make_new_decomposer(
                        self.graph_transformer.re_transform_single(new_graph))
                else:
                    self._samplelog("_propose_graph: feasibility failed",level=5)
                    continue

                # RETURN IF OK
                if new_decomposer:
                    self.compute_proposal_probability(decomposer, new_decomposer, original_cip)
                    self._samplelog(
                        "_propose_graph: step %d; core %d/%d; original_cips tried  %d; size %d; last_score %.4f" %
                        (self.step, attempt, choices, orig_cip_ctr, decomposer._base_graph.number_of_nodes(),decomposer._score) )
                    new_decomposer.clean()  # i clean only here because i need the interface mark for reverse_dir_prob
                    return new_decomposer
                else:
                    logger.log("_propose_graph: new_decomposer was None")

                # quick skip was moved to cip_select

            logger.log(5,'_propose_graph orig_cip# %d cips_tried# %s' % (orig_cip_ctr,str(attempt)) )

        #draw.graphlearn(original_cip,self.lsgg.productions[original_cip].vlues())


    def compute_proposal_probability(self, decomposer, decomposer_new, cip):
        """t
        MCMC required that the state change has the same probability back and forth.
        Going from one graph to another might alter the new graph such that many new destinations are available.
        If you want to use MCMC sampling, we should account for that.

        This function estimates the effect such that the acceptance function can use this estimation.

        The estimation is based on the "jump possibilities" of all the nodes in the core before and after the
        substitution.

        Parameters
        ----------
        decomposer: nx.Graph
            old graph
        decomposer_new: nx.Graph
            mew graph
        cip: CoreInterfacePair
            the old cip is enough since we mainly need the ids of the interface

        Returns
        -------
            sets proposal_probability_value
        """

        def ops(decomposer, cip_graph):
            counter = 0
            interfacesize = 0
            for n, d in cip_graph.nodes(data=True):
                if 'edge' not in d and 'interface' in d:
                    cips = decomposer.rooted_core_interface_pairs(n, radius_list=self.lsgg.radius_list,
                                                                  thickness_list=self.lsgg.thickness_list)
                    for cip in cips:
                        if cip.interface_hash in self.lsgg.productions:
                            counter += len(self.lsgg.productions[cip.interface_hash])
                    interfacesize += 1
            if interfacesize == 0:
                raise Exception('calc_proposal_probability: the proposed graph doesn\'t \
                    know which nodes were interfaces before; sampler parameters let you deactivate this operation')
            return counter, interfacesize

        if self.proposal_probability:
            old_opts, interfacesize = ops(decomposer, cip.graph)
            new_opts, unused = ops(decomposer_new, decomposer_new.base_graph())
            average_opts = float(old_opts + new_opts) / 2
            old_opts = max(1, old_opts)
            new_opts = max(1, new_opts)
            v1 = new_opts + average_opts * (len(decomposer_new.base_graph()) - interfacesize)
            v2 = old_opts + average_opts * (len(decomposer.base_graph()) - interfacesize)
            value = float(v1) / v2
            self.proposal_probability_value = value
            self._samplelog('reverse_direction_modifier: %f' % value, level=5)


def _sample_multi(what):
    graphlearner = dill.loads(what[0])
    graphlist = dill.loads(what[1])
    # if jobsize % batchsize != 0, sample will not give me a tuple,
    # here i filter for these
    multisampletime=time.time()
    #result = [graphlearner.transform_single(g) for g in graphlist]
    result = map(graphlearner.transform_single, graphlist)
    logger.log(5,'sampled a batch in %.2fs' % (time.time()-multisampletime))
    # print result
    return [e for e in result if type(e) == type(())]
