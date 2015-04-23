import joblib
import utils.myeden as graphlearn_utils
import networkx as nx
import itertools
import random
from multiprocessing import Pool,Manager
from eden.graph import Vectorizer
from networkx.algorithms import isomorphism as iso
from eden import fast_hash
import utils.draw as draw
import logging
import numpy
import dill
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import vstack

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')

cons = logging.StreamHandler()
cons.setLevel(logging.INFO)
cons.setFormatter(formatter)
logger.addHandler(cons)
file = logging.FileHandler('run.log',mode='w')
file.setLevel(logging.DEBUG)
file.setFormatter(formatter)
logger.addHandler(file)




"""
    the adaptiveMHgraphsamper has 3 parts:
        -init()  will be followed by
            -load() 
            -train_estimator_and_extract_grammar()
        
        -core substitution 
            in this section you can find substitution related function
            these are of no interest to you
        
        -sample_set(self,graph_iter,LOTS OF OPTIONS)
            will improve a set of graphs 'times' times each and 
            yield when done        
    
    then there is the grammar functions class to induce grammars
        probably not so interesting
    
    and we have a feasibility checker to see if generated graphs
    are valid == may be interesting to the user

    there is also the extractor for cores and interfaces
"""


class GraphLearnSampler:
    """
         wirte something
    """

    def __init__(self,radius_list=[],thickness_list=[],estimator=None,grammar=None,hash_bits=20):

        self.feasibility_checker = FeasibilityChecker()
        self.vectorizer_expanded = graphlearn_utils.GraphLearnVectorizer(complexity=3)
        self.vectorizer_normal = Vectorizer(complexity = 3)
        self.radius_list=radius_list
        self.thickness_list=thickness_list
        self.estimator=estimator
        self.local_substitutable_graph_grammar=grammar
        self.hash_bitmask=pow(2, hash_bits) - 1



    def save(self, filename):
        joblib.dump(self.__dict__, filename, compress=1)

    def load(self, filename):
        self.__dict__ = joblib.load(filename)

    def induce_grammar(self, G_iterator,core_interface_pair_remove_threshold=3,
                 interface_remove_threshold=2, n_jobs=4,):
        '''
        you might want to overwrite this to customize how the grammar is induced
        extract_cores_and_interfaces_multi
        '''
        if not self.radius_list:
            raise Exception ( "ERROR: tell me how to induce a grammar" )


        self.local_substitutable_graph_grammar = LocalSubstitutableGraphGrammar(self.radius_list, self.thickness_list,
                                                                core_interface_pair_remove_threshold,
                                                                interface_remove_threshold,
                                                                hash_bitmask=self.hash_bitmask)
        self.local_substitutable_graph_grammar.read(G_iterator, n_jobs)
        self.local_substitutable_graph_grammar.clean()


    def fit_estimator(self, iterable_pos_train, n_jobs=-1, cv=10):
        # i think this works on normal graphs..
        X_pos_train = self.vectorizer_normal.transform(iterable_pos_train)
        X_neg_train = X_pos_train.multiply(-1)
        # optimize hyperparameters classifier
        estimator = graphlearn_utils.graphlearn_fit_estimator(positive_data_matrix=X_pos_train,
                                                  negative_data_matrix=X_neg_train,
                                                  cv=cv,
                                                  n_jobs=n_jobs)

        CUTOF=.1
        # l = [self.estimator.decision_function(g) for g in X_pos_train]
        l = [(estimator.predict_proba(g)[0][1],g)   for g in X_pos_train]
        l.sort(key= lambda x:x[0])
        element = int(len(l) * CUTOF)
        estimator.intercept_ -= l[element][0]


        # calibrate
        data_matrix=vstack( [ a[1] for a in l  ] )
        data_y=numpy.asarray([0]*element +[1]*(len(l)-element))
        self.estimator = CalibratedClassifierCV(estimator, cv=2, method='sigmoid')
        self.estimator.fit(data_matrix,data_y  )




    def fit(self, G_pos,
            core_interface_pair_remove_threshold=3,
            interface_remove_threshold=2,
            n_jobs=-1):

        G_iterator, G_iterator_ = itertools.tee(G_pos)

        self.induce_grammar(G_iterator,
                            core_interface_pair_remove_threshold,
                            interface_remove_threshold,
                            n_jobs)

        self.fit_estimator(G_iterator_, n_jobs)


    ###########################  core substitution things ####################

    def merge(self, G, node, node2):
        '''
        all nodes are strings, node is the king
        '''
        for n in G.neighbors(node2):
            G.add_edge(node, n)
        G.node[node]['interface'] = True
        G.remove_node(node2)


    def find_isomorphism(self, home, other):
        matcher = lambda x, y: x['label'] == y['label']
        GM = iso.GraphMatcher(home, other, node_match=matcher)
        if GM.is_isomorphic() == False:
            return {}
        return GM.mapping


    def core_substitution(self, graph, subgraph, newcip_graph):
        """
        graph is the whole graph..
        subgraph is the interfaceregrion in that we will transplant
        newcip_graph which is the interface and the new core
        """
        # select only the interfaces of the cips
        nocore = [n for n, d in newcip_graph.nodes(data=True) if d.has_key('core') == False]
        newgraph_interface = nx.subgraph(newcip_graph, nocore)
        nocore = [n for n, d in subgraph.nodes(data=True) if d.has_key('core') == False]
        subgraph_interface = nx.subgraph(subgraph, nocore)
        # get isomorphism between interfaces, if none is found we return an empty graph
        iso = self.find_isomorphism(subgraph_interface, newgraph_interface)
        if len(iso) != len(subgraph_interface):
            return nx.Graph()
        # ok we got an isomorphism so lets do the merging
        G = nx.union(graph, newcip_graph, rename=('', '-'))
        # removing old core
        nocore = [n for n, d in subgraph.nodes(data=True) if d.has_key('core')]
        for n in nocore:
            G.remove_node(str(n))
        # merge interfaces
        for k, v in iso.iteritems():
            self.merge(G, str(k), '-' + str(v))
        # unionizing killed my labels so we need to relabel
        return nx.convert_node_labels_to_integers(G)


    ############ imporoving stuff ##################




    def grammar_preprocessing(self):
        '''
            we change the grammar according to the sampling task
        '''
        if self.n_jobs > 0:
            self.local_substitutable_graph_grammar.turn_into_managed_dict()
        if self.same_radius:
            self.local_substitutable_graph_grammar.mark_radius()



    def sample(self, graph_iter,same_radius=True,similarity=-1,sampling_interval=9999,
               batch_size=10,
               n_jobs=0,
               n_steps=50,
               postprocessing = (lambda x:x)):

        """
            create an iterator over lists.
            each list is the changing history of a start graph.
            the last graph is the final result. there is also a graph.scorehistory

            for options here you can look at the improvement_rules and you may want to ovewrite these:
                def filter_available_cips
                def choose_cip
        """
        self.same_radius= same_radius
        self.similarity=similarity
        self.sampling_interval=sampling_interval
        self.n_steps=n_steps
        self.n_jobs=n_jobs

        # multiprocessing work-around oOoOoo sadly necessary
        self.postprocessing=dill.dumps(postprocessing)

        # adapt grammar to task:
        self.grammar_preprocessing()

        # do the improvement
        if n_jobs == 0:
            for graph in graph_iter:
                yield self._sample(graph)
        else:
            # make it so that we dont need to copy the whole grammar a few times  for multiprocessing
            problems = itertools.izip(
                graph_iter, itertools.repeat(self))
            pool = Pool(processes=n_jobs)
            batch_size=batch_size
            it = pool.imap_unordered(improve_loop_multi, problems, batch_size)
            for pair in it:
                yield pair
            pool.close()




    def similarity_checker(self,graph,set_reference=False):
        '''
        always check if similarity is relevant.. if so then:

        if set_reference is True:
            remember the vectorized object
        else:
            similarity between start graph and current graph is expected to decrease.
            if similarity meassure smaller than the limit, we stop
            because we dont want to drift further
        '''
        if self.similarity>0:
            if set_reference:
                self.vectorizer_expanded._reference_vec = \
                    self.vectorizer_expanded._convert_dict_to_sparse_matrix(
                    self.vectorizer_expanded._transform(0,nx.Graph(graph)) )
            else:
                similarity = self.vectorizer_expanded._similarity(graph,[1])
                return similarity < self.similarity
        return False




    def _sample_init(self,graph):
        '''
        we prepare a single graph to be sampled

        - first we expand its edges to nodes, so eden will be able wo work its magic on it
        - then we calculate a score for the graph, to see how much we like it
        - in the end we set up one of the stop conditions for the sample loop
        '''
        graph = graphlearn_utils.expand_edges(graph)
        self.score(graph)
        self.similarity_checker(graph, set_reference=True)

        if type(self.postprocessing)==str:
            self.postprocessing = dill.loads(self.postprocessing)
        return graph


    def _sample(self, graph):
        '''
            we sample a single graph.
        '''

        # prepare variables and graph
        graph=self._sample_init(graph)
        scores = [graph.score]
        sample_path=[graph]
        accept_counter=0

        # note to attach to the result in the end.
        # currently it contains information on why we stopped the sampling before n_steps
        special_notes='None'
        for x in xrange(self.n_steps):

            # do we need to stop early??
            if self.similarity_checker(graph):
                special_notes='sample stopped; reason:similarity; at_step: '+str(x)
                break

            # do an  improvement step
            candidate_graph = self.propose(graph)
            if candidate_graph == None:
                logger.info( "sample stopped; no propose after %d successful improvement_steps" % x)
                # this will show you the failing graph:
                # draw.display(graph)
                special_notes='sample stopped; reason:no candidate found; see logfile for details; at_step: '+str(x)

                break

            graph, is_accepted = self.accept(graph,candidate_graph)
            accept_counter += is_accepted

            # save score and if needed a snapshot of the current graph
            scores.append(graph.score)
            if x % self.sampling_interval == 0:
                sample_path.append(graph)


        # return a list of graphs, the last of wich is our final result
        sample_path.append(graph)
        res=self.vectorizer_expanded._revert_edge_to_vertex_transform(graph)
        # filling the score_history so we have as n_steps many entries ,,
        # i just repeat the last entry until len(scores) is n_steps
        scores += [scores[-1]] * (self.n_steps+1 - len(scores))
        graph_info={}
        graph_info['graphs']=sample_path
        graph_info['score_history']=scores
        graph_info['accept_count']=accept_counter
        graph_info['notes']=special_notes
        return (res,graph_info)



    def score(self,graph):
        if not 'score' in graph.__dict__:
            transformed_graph = self.vectorizer_expanded.transform2(graph)
            #graph.score = self.estimator.decision_function(transformed_graph)[0]
            graph.score = self.estimator.predict_proba(transformed_graph)[0][1]
            # print graph.score
        return graph.score

    def accept(self,graph_old,graph_new):
        '''
        returns the graph that was decided to be better, and an indicator if the better one is the new one
        '''

        graph_new = self.postprocessing(graph_new)

        self.score(graph_old)
        self.score(graph_new)

        if graph_old.score==0:
            return graph_new,True
        score_ratio = graph_new.score/graph_old.score

        if score_ratio > random.random():
            return graph_new,True
        return graph_old,False



    def propose(self, graph):
        """
        starting from 'graph' we construct a novel candidate instance
        """
        # finding a legit candidate..
        selected_cip = self.select_cip_for_substitution(graph)
        if selected_cip == None:
            logger.debug( "propose failed; because choose_cip failed" )
            return


        # see which substitution to make
        candidate_cips = self.select_cips_from_grammar(selected_cip)
        cipcount=0
        for candidate_cip in candidate_cips:
            cipcount+=1
            # substitute and return

            graph_new=self.core_substitution(graph, selected_cip.graph, candidate_cip.graph)
            self.graph_clean(graph_new)
            if self.feasibility_checker.check(graph_new):
                return graph_new
            # else:
            #    draw.drawgraphs([graph, selected_cip.graph, candidate_cip.graph], contract=False)

        logger.debug( "propose failed;received %d cips, all of which failed either at substitution or feasibility  " % cipcount)

    def graph_clean(self,graph):
        for n, d in graph.nodes(data=True):
            d.pop('core', None)
            d.pop('interface', None)


    def select_cips_from_grammar(self,cip):
        core_cip_dict = self.local_substitutable_graph_grammar.grammar[cip.interface_hash]
        if core_cip_dict:
            if self.same_radius:
                hashes=self.local_substitutable_graph_grammar.radiuslookup[cip.interface_hash][cip.radius]
            else:
                hashes = core_cip_dict.keys()
            random.shuffle(hashes)
            for core_hash in hashes:
                # does the new core need to be the same size as the old one??
                if self.same_radius:

                    if core_cip_dict[core_hash].radius != cip.radius:
                        continue
                yield core_cip_dict[core_hash]
        logger.debug('select_cips_from_grammar didn\'t find any acceptable cip; entries_found %d' % len(core_cip_dict ))


    def select_cip_for_substitution(self, graph):
        """
            selects a chip randomly from the graph.
            root is a node_node and not an edge_node
            radius and thickness are chosen to fit the grammars radius and thickness
        """
        tries=20
        failcount=0
        for x in xrange(tries):
            node = random.choice(graph.nodes())
            if 'edge' in graph.node[node]:
                node = random.choice(graph.neighbors(node))
            # random radius and thickness
            radius = random.choice(  self.local_substitutable_graph_grammar.radius_list  )
            thickness = random.choice(self.local_substitutable_graph_grammar.thickness_list)

            # exteract_core_and_interface will return a list of results, we expect just one so we unpack with [0]
            # in addition the selection might fail because it is not possible to extract at the desired radius/thicknes
            #
            cip= extract_core_and_interface(node, graph, [radius], [thickness], vectorizer=self.vectorizer_normal,
                                            hash_bitmask=self.hash_bitmask)

            # if radius and thickness are not possible to extract cip is [] which is false
            if not cip:
                failcount+=1
                continue
            cip=cip[0]
            # if we have a hit in the grammar
            if cip.interface_hash in self.local_substitutable_graph_grammar.grammar:
                #  if we have the same_radius rule implemented:
                if self.same_radius:
                    # we jump if that hit has not the right radius
                    if cip.radius not in self.local_substitutable_graph_grammar.radiuslookup[cip.interface_hash]:
                        continue

                return cip

        logger.debug ('choose_cip failed because no suiting interface was found, extract failed %d times ' % (failcount))




# ok moving this here instead of leaving it where it belongs prevents pickling errar ..
#dont quite get it ...        
def improve_loop_multi(x):
    return x[1]._sample(x[0])


################ALL THE THINGS HERE SERVE TO LEARN A GRAMMAR ############



class core_interface_pair:
    def __init__(self, ihash=0, chash=0, graph=0, radius=0, thickness=0,core_nodes_count=0):
        self.interface_hash = ihash
        self.core_hash = chash
        self.graph = graph
        self.radius = radius
        self.thickness = thickness
        self.core_nodes_count=core_nodes_count





class LocalSubstitutableGraphGrammar:
    """
    i had this class inherit from default dict, but that breaks joblib oOo
    and i cant load anymore.
    """
    # move all the things here that are needed to extract grammar
    def __init__(self, radius_list, thickness_list, core_interface_pair_remove_threshold=3,
                 interface_remove_threshold=2,hash_bitmask=2**20-1):
        self.grammar = {}
        self.interface_remove_threshold = interface_remove_threshold
        self.radius_list = radius_list
        self.thickness_list = thickness_list
        self.core_interface_pair_remove_threshold = core_interface_pair_remove_threshold
        self.vectorizer = graphlearn_utils.GraphLearnVectorizer()
        self.hash_bitmask=hash_bitmask



    def turn_into_managed_dict(self):
        '''
        this turns the grammar into a managed dictionary which we need for multiprocessing

        note that we dont do this per default because then we cant save the grammar anymore
        while keeping the manager outside the object
        '''
         # move the grammar into a manager object...
        manager = Manager()
        shelve = manager.dict()
        for k, v in self.grammar.iteritems():
            md=manager.dict()
            for k2,v2 in v.iteritems():
                md[k2]=v2
            shelve[k] = md
        self.grammar = shelve

    def difference(self,other_grammar,substract_cip_count=False):

        # i assume set() my even be faster than sort...
        # should be even faster to just test -_-

        #mykeys=set(my.keys())
        #otherkeys=set(other_grammar.keys())
        for ihash in self.grammar.keys():
            if ihash in other_grammar:
                # so if my ihash is in both i can compare the corehashes
                for chash in self.grammar[ihash].keys():
                    if chash in other_grammar[ihash]:
                        if substract_cip_count:
                            self.grammar[ihash][chash].count-=self.grammar[ihash][chash].count
                        else:
                            self.grammar[ihash].pop(chash)
        if substract_cip_count:
            self.clean()




    def union(self, other_grammar, add_cip_count=False):
        '''
        we build the union of grammars..
        '''

        for ihash in other_grammar.keys():
            if ihash in self.grammar:
                # ok so we are in the intersection...
                for chash in other_grammar[ihash]:
                    if chash in self.grammar[ihash] and add_cip_count:
                        self.grammar[ihash][chash].counter+=other_grammar[ihash][chash].counter
                    else:
                        self.grammar[ihash][chash]=other_grammar[ihash][chash]
            else:
                self.grammar[ihash]=other_grammar[ihash]

    def intersect(self,other_grammar):
        # counts will be added...

        for ihash in self.grammar.keys():
            if ihash in other_grammar:
                # so if self.grammar ihash is in both i can compare the corehashes
                for chash in self.grammar[ihash].keys():
                    if chash in other_grammar[ihash]:
                        self.grammar[ihash][chash].counter+= other_grammar[ihash][chash].counter
                    else:
                        self.grammar[ihash].pop(chash)
            else:
                self.grammar.pop(ihash)


    def clean(self):
        """
            after adding many cips to the grammar it is possible
            that some thresholds are not reached, and we dont want unnecessary ballast in
            our grammar.
            so we clean up.
        """
        for interface in self.grammar.keys():

            for core in self.grammar[interface].keys():
                if self.grammar[interface][core].count < self.core_interface_pair_remove_threshold:
                    self.grammar[interface].pop(core)

            if len(self.grammar[interface]) < self.interface_remove_threshold:
                self.grammar.pop(interface)

    def mark_radius(self):
        '''
            there is now self.radiuslookup{ interfacehash: {radius:[list of corehashes with that ihash and radius]} }
        '''
        self.radiuslookup={}
        for interface in self.grammar.keys():
            radiuslookup={}
            for core in self.grammar[interface].keys():
                radius=self.grammar[interface][core].radius
                if radius in radiuslookup:
                    radiuslookup[radius].append(core)
                else:
                    radiuslookup[radius]=[core]
            self.radiuslookup[interface]=radiuslookup





    def read(self, graphs, n_jobs=4):
        '''
        we extract all chips from graphs of a graph iterator
        we use n_jobs processes to do so.
        '''


        # if we should use only one process we use read_single ,  else read_multi
        if n_jobs == 1:
            self.read_single(graphs)
        else:
            self.read_multi(graphs, n_jobs)

    def grammar_add_core_interface_data(self, cid):

        '''
            cid is a core interface data instance.
            we will add the cid to our grammar.
        '''


        # initialize gramar[interfacehash] if necessary
        if cid.interface_hash not in self.grammar:
            self.grammar[cid.interface_hash] = {}

        # initialize or get grammar[interfacehash][corehash] which is now called subgraph_data
        if cid.core_hash in self.grammar[cid.interface_hash]:
            subgraph_data = self.grammar[cid.interface_hash][cid.core_hash]
        else:
            subgraph_data = core_interface_pair()
            self.grammar[cid.interface_hash][cid.core_hash] = subgraph_data
            subgraph_data.count=0

        # put new information in the subgraph_data
        # we only save the count until we know that we will keep the actual cip
        subgraph_data.count += 1
        if subgraph_data.count == self.core_interface_pair_remove_threshold:
            subgraph_data.__dict__.update(cid.__dict__)
            subgraph_data.count = self.core_interface_pair_remove_threshold

    def read_single(self, graphs):
        """
            for graph in graphs:
                get cips of graph
                    put cips into grammar
        """
        for gr in graphs:
            problem = (gr, self.radius_list,self.thickness_list, self.vectorizer,self.hash_bitmask)
            for core_interface_data_list in extract_cores_and_interfaces(problem):
                for cid in core_interface_data_list:
                    self.grammar_add_core_interface_data(cid)

    def read_multi(self, graphs, n_jobs):
        """
        will take graphs and to multiprocessing to extract their cips
        and put these cips in the grammar
        """

        # generate iterator of problem instances
        problems = itertools.izip(graphs, itertools.repeat(self.radius_list),
                                  itertools.repeat(self.thickness_list),
                                  itertools.repeat(self.vectorizer),
                                  itertools.repeat(self.hash_bitmask))

        # creating pool of workers
        pool = Pool(processes=n_jobs)
        # distributing jobs to workers
        result = pool.imap_unordered(extract_cores_and_interfaces, problems, 10)
        # the resulting chips can now be put intro the grammar
        for core_interface_data_listlist in result:
            for core_interface_data_list in core_interface_data_listlist:
                for cid in core_interface_data_list:
                    self.grammar_add_core_interface_data(cid)

        pool.close()




def extract_cores_and_interfaces(x):
    # unpack arguments, expand the graph
    graph, radius_list, thickness_list, vectorizer,hash_bitmask = x
    graph = graphlearn_utils.expand_edges(graph)
    ret = []
    for node in graph.nodes_iter():
        if 'edge' in graph.node[node]:
            continue
        core_interface_list = extract_core_and_interface(node, graph, radius_list, thickness_list,
                                                    vectorizer=vectorizer,hash_bitmask=hash_bitmask)
        if core_interface_list:
            ret.append(core_interface_list)
    return ret


# ############################### FEASIBILITY CHECKER ###################


class FeasibilityChecker():
    def __init__(self):
        self.checklist = []
        self.checklist.append(defaultcheck)

    def check(self, ng):
        for f in self.checklist:
            if f(ng) == False:

                return False
        return True


def defaultcheck(ng):
    if len(ng) < 1:
        logger.debug( 'graph non existent' )
        return False
    for node_id in ng.nodes_iter():
        if 'edge' in ng.node[node_id]:
            if len(ng.neighbors(node_id)) != 2:
                return False
    return True




#####################################   extract a core/interface pair #####################









def inversedict(d):
    d2 = {}
    for k, v in d.iteritems():
        l=[]
        d2[v] = d2.get(v, l)
        d2[v].append(k)
    return d2


def calc_interface_hash(interface_graph, hash_bitmask):
    l = []
    node_name_cache = {}

    all_nodes=set( interface_graph.nodes() )
    visited=set()
    # all the edges
    for (a, b) in interface_graph.edges():
        visited.add(a)
        visited.add(b)

        ha = node_name_cache.get(a, -1)
        if ha == -1:
            ha = calc_node_name(interface_graph, a,hash_bitmask)
            node_name_cache[a] = ha
        hb = node_name_cache.get(b, -1)
        if hb == -1:
            hb = calc_node_name(interface_graph, b,hash_bitmask)
            node_name_cache[b] = hb
        l.append((ha ^ hb) + (ha + hb))
    l.sort()

    # nodes that dont have edges
    l+=[  interface_graph.node[node_id]['hlabel'][0]  for node_id in  all_nodes-visited ]
    l = fast_hash(l,hash_bitmask)
    return l



def calc_core_hash(core_graph,hash_bitmask):
    return calc_interface_hash(core_graph, hash_bitmask)



def calc_node_name(interfacegraph, node,hash_bitmask):
    d = nx.single_source_shortest_path_length(interfacegraph, node, 20)
    # d is now node:dist
    # l is a list of  hash(label,distance)
    #l=[   func([interfacegraph.node[nid]['intlabel'],dis])  for nid,dis in d.items()]
    l = [interfacegraph.node[nid]['hlabel'][0] + dis for nid, dis in d.items()]
    l.sort()
    l = fast_hash(l,hash_bitmask)
    return l


def extract_core_and_interface(node, graph, radius_list=None, thickness_list=None, vectorizer=None,
                               hash_bitmask=2*20-1):

    if 'hlabel' not in graph.node[0]:
        vectorizer._label_preprocessing(graph)

    # which nodes are in the relevant radius
    dist = nx.single_source_shortest_path_length(graph, node, max(radius_list) + max(thickness_list))
    # we want the relevant subgraph and we want to work on a copy
    retgraph = nx.Graph(graph.subgraph(dist))

    # we want to inverse the dictionary.
    # so now we see {distance:[list of nodes at that distance]}
    nodedict = inversedict(dist)


    cip_list = []
    for thickness_ in thickness_list:
        for radius_ in radius_list:


            # see if it is feasable to extract
            if radius_ + thickness_ not in nodedict:
                continue

            #calculate hashes
            #d={1:[1,2,3],2:[3,4,5]}
            #print [ i for x in [1,2] for i in d[x] ]
            interface_graph_nodes = [item for x in range(radius_ + 1, radius_ + thickness_ + 1) for item in
                                     nodedict.get(x, [])]
            interfacehash = calc_interface_hash(retgraph.subgraph(interface_graph_nodes), hash_bitmask)

            core_graph_nodes = [item for x in range(radius_ + 1) for item in nodedict.get(x, [])]
            corehash = calc_core_hash(retgraph.subgraph(core_graph_nodes),hash_bitmask)

            #get relevant subgraph
            nodes = [node for i in range(radius_ + thickness_ + 1) for node in nodedict[i]]
            cip_graph = nx.Graph(retgraph.subgraph(nodes))

            #marking cores and interfaces in subgraphs
            for i in range(radius_ + 1):
                for no in nodedict[i]:
                    cip_graph.node[no]['core'] = True
                    if 'interface' in cip_graph.node[no]:
                        cip_graph.node[no].pop('interface')
            for i in range(radius_ + 1, radius_ + thickness_ + 1):
                if i in nodedict:
                    for no in nodedict[i]:
                        cip_graph.node[no]['interface'] = True
                        if 'core' in cip_graph.node[no]:
                            cip_graph.node[no].pop('core')

            core_nodes_count=sum([len(nodedict[x]) for x in range(radius_)])
            cip_list.append(core_interface_pair(interfacehash, corehash, cip_graph, radius_, thickness_,core_nodes_count))
    return cip_list


