import joblib
import utils.myeden as graphlearn_utils
from eden.util import fit_estimator as eden_fit_estimator
import networkx as nx
import itertools
import random
from multiprocessing import Pool
from eden.graph import Vectorizer
import utils.draw as draw
import logging
import numpy
import dill
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
import postprocessing


from graphtools import extract_core_and_interface, core_substitution
from feasibility import FeasibilityChecker
from grammar import LocalSubstitutableGraphGrammar

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
cons = logging.StreamHandler()
cons.setLevel(logging.INFO)
cons.setFormatter(formatter)
logger.addHandler(cons)
file = logging.FileHandler('run.log', mode='w')
file.setLevel(logging.DEBUG)
file.setFormatter(formatter)
logger.addHandler(file)


class GraphLearnSampler:

    def __init__(self, radius_list=[3, 5], thickness_list=[2, 4], estimator=None, grammar=None, nbit=20,
                 vectorizer= graphlearn_utils.GraphLearnVectorizer(complexity=3),

                 ):



        self.feasibility_checker = FeasibilityChecker()
        self,postprocessor = postprocessing.postprocessor()

        # see utils.myeden.GraphLeanVectorizer,
        # edens vectorizer assumes that graphs are not expanded.
        # this is fixed with just a few lines of code.
        self.vectorizer = vectorizer


        # lists of int
        self.radius_list = radius_list
        self.thickness_list = thickness_list
        # scikit  classifier
        self.estimator = estimator
        # grammar object
        self.local_substitutable_graph_grammar = grammar
        # cips hashes will be masked with this
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

    def save(self, file_name):
        self.local_substitutable_graph_grammar.revert_multicore_transform()
        joblib.dump(self.__dict__, file_name, compress=1)

    def load(self, file_name):
        self.__dict__ = joblib.load(file_name)






    def fit_estimator(self, X, n_jobs=-1, cv=10):
        '''
        create self.estimator...
        by inversing the X set to get a negative set
        and then using edens fit_estimator
        '''

        # get negative set:
        X_neg = X.multiply(-1)
        # i hope loss is log.. not 100% sure..
        # probably calibration will fix this#
        self.estimator = eden_fit_estimator(SGDClassifier(), positive_data_matrix=X,
                                            negative_data_matrix=X_neg,
                                            cv=cv,
                                            n_jobs=n_jobs,
                                            verbose=0,
                                            n_iter_search=20)

    def calibrate_estimator(self, X, nu=.5):
        '''
            move bias until nu of X are in the negative class

            then use scikits calibrate to calibrate self.estimator arround the input
        '''
        #  move bias
        l = [(self.estimator.decision_function(g)[0], g) for g in X]
        l.sort(key=lambda x: x[0])
        element = int(len(l) * nu)
        self.estimator.intercept_ -= l[element][0]

        # calibrate
        data_matrix = vstack([a[1] for a in l])
        data_y = numpy.asarray([0] * element + [1] * (len(l) - element))
        self.estimator = CalibratedClassifierCV(self.estimator, cv=3, method='sigmoid')
        self.estimator.fit(data_matrix, data_y)

    def fit(self, G_pos,
            core_interface_pair_remove_threshold=3,
            interface_remove_threshold=2,
            n_jobs=-1, nu=.5):
        """
          use input to fit the grammar and fit the estimator
        """
        G_iterator, G_iterator_ = itertools.tee(G_pos)


        self.local_substitutable_graph_grammar = LocalSubstitutableGraphGrammar(self.radius_list, self.thickness_list,
                                                                                core_interface_pair_remove_threshold,
                                                                                interface_remove_threshold,
                                                                                nbit=self.nbit)
        self.local_substitutable_graph_grammar.fit(G_iterator,n_jobs)


        X = self.vectorizer.transform(G_iterator_)
        self.fit_estimator(X, n_jobs)
        self.calibrate_estimator(X, nu)



    ############################### SAMPLE ###########################

    def grammar_preprocessing(self):
        '''
            we change the grammar according to the sampling task
        '''
        if self.n_jobs > 0:
            self.local_substitutable_graph_grammar.multicore_transform()
        if self.same_radius:
            self.local_substitutable_graph_grammar.add_same_radius_quicklookup()
        if self.same_core_size:
            self.local_substitutable_graph_grammar.add_core_size_quicklookup()

    def sample(self, graph_iter, same_radius=False, same_core_size=True, similarity=-1, sampling_interval=9999,
               batch_size=10,
               n_jobs=0,
               n_steps=50
               ):
        """
            input: graph iterator
            output: yield (sampled_graph,{dictionary of info about sampling process}
        """
        self.same_radius = same_radius
        self.similarity = similarity
        self.sampling_interval = sampling_interval
        self.n_steps = n_steps
        self.n_jobs = n_jobs
        self.same_core_size = same_core_size


        # adapt grammar to task:
        self.grammar_preprocessing()

        # do the improvement
        if n_jobs in [0, 1]:
            for graph in graph_iter:
                yield self._sample(graph)
        else:
            # make it so that we dont need to copy the whole grammar a few times  for multiprocessing
            problems = itertools.izip(
                graph_iter, itertools.repeat(self))

            if n_jobs > 1:
                pool = Pool(processes=n_jobs)
            else:
                pool = Pool()


            #it = pool.imap_unordered(improve_loop_multi, problems, batch_size)
            #_sample_multi=lambda x: x[1]._sample(x[0])

            it = pool.imap_unordered(_sample_multi, problems, batch_size)


            for pair in it:
                yield pair
            pool.close()

    def _sample(self, graph):
        '''
            we sample a single graph.

            input: a graph
            output: (sampled_graph,{info dictionary})
        '''

        # prepare variables and graph
        graph = self._sample_init(graph)
        scores_log = [graph.score]
        scores = [graph.score_nonlog]
        sample_path = [graph]
        accept_counter = 0

        for step in xrange(self.n_steps):
            # do an  improvement step
            candidate_graph = self.propose(graph)
            # can we stop now?
            if self._sampling_stop_condition(graph, candidate_graph, scores, scores_log, step):
                break

            # is the new graph better than the old?
            candidate_graph = self.postprocessor.postprocess(candidate_graph)
            if self.accept(graph, candidate_graph):
                accept_counter += 1
                graph = candidate_graph
            # save score
            # take snapshot
            scores.append(graph.score_nonlog)
            scores_log.append(graph.score)
            if step % self.sampling_interval == 0:
                sample_path.append(graph)

        # we put the result in the sample_path
        # and we return a nice graph as well as a dictionary of additional information
        sample_path.append(graph)
        sampled_graph = self.vectorizer._revert_edge_to_vertex_transform(graph)
        sampled_graph_info =  {'graphs': sample_path, 'score_history': scores, "log_score_history": scores_log, "accept_count": accept_counter, 'notes': self._sample_notes}
        return (sampled_graph, sampled_graph_info)

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
        self.score(graph)
        self.similarity_checker(graph, set_reference=True)
        self._sample_notes = ''

        if type(self.postprocessing) == str:
            self.postprocessing = dill.loads(self.postprocessing)
        return graph

    def similarity_checker(self, graph, set_reference=False):
        '''
        always check if similarity is relevant.. if so then:

        if set_reference is True:
            remember the vectorized object
        else:
            similarity between start graph and current graph is expected to decrease.
            if similarity meassure smaller than the limit, we stop
            because we dont want to drift further
        '''
        if self.similarity > 0:
            if set_reference:
                self.vectorizer._reference_vec = \
                    self.vectorizer._convert_dict_to_sparse_matrix(
                        self.vectorizer._transform(0, nx.Graph(graph)))
            else:
                similarity = self.vectorizer._similarity(graph, [1])
                return similarity < self.similarity
        return False

    def _sampling_stop_condition(self, graph, candidate_graph, scores, scores_log, step):
        '''
            we look at the status of sampling
            and decide if stop conditions are met.

            in case they are:
                we also make sure the score_histories match with the number of sample steps
        '''

        stop = False
        # do we need to stop early??
        if self.similarity_checker(graph):
            self._sample_notes = 'sample stopped; reason:similarity; at_step: ' + str(step)
            stop = True

        if candidate_graph == None:
            logger.info("sample stopped; no propose after %d successful improvement_steps" % step)
            # this will show you the failing graph:
            # draw.display(graph)
            self._sample_notes = 'sample stopped; reason:no candidate found; see logfile for details; at_step: ' + str(step)
            stop = True

        if stop:  # make sure score list is as long as n_steps
            scores += [scores[-1]] * (self.n_steps + 1 - len(scores))
            scores_log += [scores_log[-1]] * (self.n_steps + 1 - len(scores_log))

        return stop

    def score(self, graph):
        """

        :param graph: a graph
        :return: score of graph

        we also set graph.score_nonlog and graph.score

        """
        if not 'score' in graph.__dict__:
            transformed_graph = self.vectorizer.transform2(graph)
            graph.score_nonlog = self.estimator.base_estimator.decision_function(transformed_graph)[0]
            graph.score = self.estimator.predict_proba(transformed_graph)[0][1]
            # print graph.score
        return graph.score

    def accept(self, graph_old, graph_new):
        '''
            return true if graph_new scores higher
        '''

        score_graph_old = self.score(graph_old)
        score_graph_new = self.score(graph_new)

        if score_graph_old == 0:
            return True

        score_ratio = score_graph_new / score_graph_old

        if score_ratio > random.random():
            return True
        return False

    def propose(self, graph):
        """
        starting from 'graph' we construct a novel candidate instance
        return None and a debug log if we fail to do so.

        """
        # finding a legit candidate..
        selected_cip = self.select_cip_for_substitution(graph)
        if selected_cip == None:
            logger.debug("propose failed; because select_cip_for_substitution failed")
            return

        # see which substitution to make
        candidate_cips = self.select_cips_from_grammar(selected_cip)
        cipcount = 0
        for candidate_cip in candidate_cips:
            cipcount += 1
            # substitute and return

            graph_new = core_substitution(graph, selected_cip.graph, candidate_cip.graph)
            self.graph_clean(graph_new)

            if self.feasibility_checker.check(graph_new):
                return graph_new
            # ill leave this here.. use it in case things go wrong oo
            #    draw.drawgraphs([graph, selected_cip.graph, candidate_cip.graph], contract=False)

        logger.debug("propose failed;received %d cips, all of which failed either at substitution or feasibility  " % cipcount)

    def graph_clean(self, graph):
        '''
        in the precess of creating a new graph,
        we marked the nodes that were used as interface and core.
        here we remove the marks.
        :param graph:
        :return:
        '''
        for n, d in graph.nodes(data=True):
            d.pop('core', None)
            d.pop('interface', None)

    def select_cips_from_grammar(self, cip):
        """
        :param cip: the cip we selected from the graph
        :yields: cips found in the grammar that can replace the input cip

        log to debug on fail
        """
        core_cip_dict = self.local_substitutable_graph_grammar.grammar[cip.interface_hash]
        if core_cip_dict:
            if self.same_radius:
                hashes = self.local_substitutable_graph_grammar.radiuslookup[cip.interface_hash][cip.radius]
            elif self.same_core_size:
                hashes = self.local_substitutable_graph_grammar.core_size[cip.interface_hash][cip.core_nodes_count]
            else:
                hashes = core_cip_dict.keys()
            random.shuffle(hashes)
            for core_hash in hashes:
                yield core_cip_dict[core_hash]
        logger.debug('select_cips_from_grammar didn\'t find any acceptable cip; entries_found %d' % len(core_cip_dict))

    def select_cip_for_substitution(self, graph):
        """
            selects a chip randomly from the graph.
            root is a node_node and not an edge_node
            radius and thickness are chosen to fit the grammars radius and thickness
        """
        tries = 20
        failcount = 0
        for x in xrange(tries):
            node = random.choice(graph.nodes())
            if 'edge' in graph.node[node]:
                node = random.choice(graph.neighbors(node))
            # random radius and thickness
            radius = random.choice(self.local_substitutable_graph_grammar.radius_list)
            thickness = random.choice(self.local_substitutable_graph_grammar.thickness_list)

            # exteract_core_and_interface will return a list of results, we expect just one so we unpack with [0]
            # in addition the selection might fail because it is not possible to extract at the desired radius/thicknes
            #
            cip = extract_core_and_interface(node, graph, [radius], [thickness], vectorizer=self.vectorizer,
                                             hash_bitmask=self.hash_bitmask)

            # if radius and thickness are not possible to extract cip is [] which is false
            if not cip:
                failcount += 1
                continue
            cip = cip[0]
            # if we have a hit in the grammar
            if cip.interface_hash in self.local_substitutable_graph_grammar.grammar:
                #  if we have the same_radius rule implemented:
                if self.same_radius:
                    # we jump if that hit has not the right radius
                    if not self.local_substitutable_graph_grammar.radiuslookup[cip.interface_hash][cip.radius]:
                        continue
                elif self.same_core_size:
                    if cip.core_nodes_count not in self.local_substitutable_graph_grammar.core_size[cip.interface_hash]:
                        continue

                return cip

        logger.debug('select_cip_for_substitution failed because no suiting interface was found, extract failed %d times ' % (failcount))


# ok moving this here instead of leaving it where it belongs prevents pickling errar ..
# dont quite get it ...
def _sample_multi(x):
    return x[1]._sample(x[0])
