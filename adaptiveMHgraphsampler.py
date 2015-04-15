import joblib
import utils.myeden as myutils
import extract
import networkx as nx
import itertools
import random
import math
from multiprocessing import Pool,Manager
from eden.graph import Vectorizer
from networkx.algorithms import isomorphism as iso


"""
    the adaptiveMHgraphsamper has 3 parts:
        -init()  will be followed by
            -load() 
            -train_estimator_and_extract_grammar()
        
        -core substitution 
            in this section you can find substitution related function
            these are of no interest to you
        
        -mass_improve_random(self,graph_iter,times=20)
            will improve a set of graphs 'times' times each and 
            yield when done        
    
    then there is the grammar functions class to induce grammars
        probably not so interesting
    
    and we have a feasibility checker to see if generated graphs
    are valid == may be interesting to the user
    
"""


class adaptiveMHgraphsampler:
    """
         wirte something
    """

    def __init__(self):
        self.bitmask = pow(2, 20) - 1
        self.feasibility_checker = feasibility_checker()
        self.vectorizer_expanded = myutils.my_vectorizer(complexity=3)
        self.vectorizer_normal = Vectorizer(complexity = 3)

    def save(self, filename):
        joblib.dump(self.__dict__, filename, compress=1)

    def load(self, filename):
        self.__dict__ = joblib.load(filename)

    def induce_grammar(self, G_iterator, radius_list, thickness_list, n_jobs=4):
        '''
        you might want to overwrite this to customize how the grammar is induced
        extract_cores_and_interfaces_multi
        '''
        self.substitute_grammar = {}
        self.grammar_functions = local_substitutable_graph_grammar(self.substitute_grammar,
                                                                radius_list, thickness_list)
        self.grammar_functions.readgraphs(G_iterator, n_jobs)
        self.grammar_functions.clean()


    def train_estimator(self, iterable_pos_train, n_jobs=-1, cv=10):
        # i think this works on normal graphs..
        X_pos_train = self.vectorizer_normal.transform(iterable_pos_train, n_jobs=n_jobs)
        X_neg_train = X_pos_train.multiply(-1)
        # optimize hyperparameters classifier
        self.estimator = myutils.my_fit_estimator(positive_data_matrix=X_pos_train, negative_data_matrix=X_neg_train,
                                                      cv=cv, n_jobs=n_jobs)
        l = [self.estimator.decision_function(g) for g in X_pos_train]
        l.sort()
        element = int(len(l) * .1)
        self.estimator.intercept_ -= l[element]
        return self.estimator


    def train_estimator_and_extract_grammar(self, G_pos, radius_list, thickness_list, n_jobs=-1):
        G_iterator, G_pos = itertools.tee(G_pos)
        self.induce_grammar(G_iterator, radius_list, thickness_list, n_jobs)
        self.train_estimator(G_pos, n_jobs)


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




    def mass_improve_random(self, graph_iter, improvement_rules=
            {'sameradius': 1,
             'similarity':0.0,
             'snap_interval':20,
             'batch_size':10,
             'n_jobs':0,
             'improvement_steps':20,
             'get_candidate_maxtries':20,
             'postprocessing':(lambda grap:grap)
             }):
        """
            create an iterator over lists.
            each list is the changing history of a start graph.
            the last graph is the final result. there is also a graph.scorehistory

            for options here you can look at the improvement_rules and you may want to ovewrite these:
                def filter_available_cips
                def choose_cip
        """
        # setting some defaults for the improvement...
        self.improvement_rules = improvement_rules
        if 'snap_interval' not in self.improvement_rules:
            self.improvement_rules['snap_interval']=9999999
        if 'batch_size' not in self.improvement_rules:
            self.improvement_rules['batch_size']=10
        if 'get_candidate_maxtries' not in self.improvement_rules:
            self.improvement_rules['get_candidate_maxtries'] = 20
        if 'improvement_steps' not in self.improvement_rules:
            self.improvement_rules['improvement_steps']=20
        n_jobs=self.improvement_rules.get('n_jobs',0)


        # move the grammar into a manager object...
        manager = Manager()
        shelve = manager.dict()
        for k, v in self.substitute_grammar.iteritems():
            md=manager.dict()
            for k2,v2 in v.iteritems():
                md[k2]=v2
            shelve[k] = md
        self.substitute_grammar = shelve

        # do the improvement
        if n_jobs == 0:
            for graph in graph_iter:
                yield self.improve_loop(graph)
        else:
            problems = itertools.izip(
                graph_iter, itertools.repeat(self))
            pool = Pool(processes=n_jobs)
            batch_size=self.improvement_rules['batch_size']
            it = pool.imap_unordered(improve_loop_multi, problems, batch_size)
            for liste in it:
                yield liste


    def improve_loop(self, graph):
        # prepare variables and graph
        scores = []
        graph = myutils.expand_edges(graph)
        self.vectorizer_expanded._label_preprocessing(graph)
        score = -3
        retlist=[graph]
        sim = -1
        if 'similarity' in self.improvement_rules:
            sim = self.improvement_rules['similarity']
            self.vectorizer_expanded._reference_vec = self.vectorizer_expanded._convert_dict_to_sparse_matrix( self.vectorizer_expanded._transform(0,nx.Graph(graph)) )


        # ok lets go round and round
        for x in xrange(self.improvement_rules['improvement_steps']):
            # do we need to stop early??
            if sim != -1:
                if self.vectorizer_expanded._similarity(graph) < sim:
                    scores += [score] * (self.improvement_rules['improvement_steps'] - x)
                    break
            # do an  improvement step
            graph, score = self.improve_random(graph, score)
            # save score and if needed a snapshot of the current graph
            scores.append(score)
            graph.scorehistory = scores
            if x % self.improvement_rules['snap_interval'] == 0:
                retlist.append(graph)
        # return a list of graphs, the last of wich is our final result
        retlist.append(self.vectorizer_expanded._revert_edge_to_vertex_transform(graph))
        return retlist

    def improve_random(self, graph, oldscore, debug=1 ):
        """
        graph is now expanded
        debug will remove core/interface attributes of nodes
            this is not necessary but makes the graph
            prettier to print
        """

        # do we need to remove signs of usage from the graph?
        if debug > 0:
            for n, d in graph.nodes(data=True):
                d.pop('core', None)
                d.pop('interface', None)


        # perform checks to see if candidate is something legit
        # postprocess
        # feasibility check
        # vectorize
        candidate = self.propose_candidate(graph)
        if len(candidate) == 0:
            if debug > 1:
                print "no candidate"
            return graph, oldscore

        if 'postprocessig' in self.improvement_rules:
            candidate = self.improvement_rules['postprocessig'](candidate)
        if not self.feasibility_checker.check(candidate):
            return graph, oldscore
        try:
            transformed_graph = self.vectorizer_expanded.transform2(candidate)
        except:
            if debug > 1:
                print "transformation failed"
            return graph, oldscore



        # decide if we keep the candidate...
        value = self.estimator.decision_function(transformed_graph)[0]
        kfactor = 15
        v1 = 1.0 / (1 + math.exp(-oldscore * kfactor))
        v2 = 1.0 / (1 + math.exp(-value * kfactor))
        randf = random.random()
        if v2 / v1 > randf:
            return candidate, value
        return graph, oldscore


    def propose_candidate(self, graph):
        """
        graph is now expanded
        """
        for tries in xrange(self.improvement_rules['get_candidate_maxtries']):
            # finding a legit candidate... and check if we can substitute anything...
            candidate = self.choose_cip(graph)
            if not candidate:
                continue
            candidate = candidate[0]
            if candidate.interface_hash not in self.substitute_grammar:
                continue

            # for each possible new core:
            for cip in self.filter_available_cips(candidate.interface_hash):
                # does the new core need to be the same size as the old one??
                if 'sameradius' in self.improvement_rules:
                    if cip.radius != candidate.radius:
                        continue

                # did we do exactly this replacement before?
                # if so it had been rejected.. dont do it again!
                substitution_hash = candidate.core_hash ^ cip.core_hash ^ candidate.interface_hash
                if 'tried_and_failed_substitutions' not in graph.__dict__:
                    graph.tried_and_failed_substitutions=[]
                elif substitution_hash in graph.tried_and_failed_substitutions:
                    continue
                graph.tried_and_failed_substitutions.append(substitution_hash)

                # return if substitution is success!
                ng = self.core_substitution(graph, candidate.graph, cip.graph)
                if len(ng) > 1:
                    return ng
        return nx.Graph()

    def filter_available_cips(self,new_interface_hash):
            core_cip_dict = self.substitute_grammar.get[new_interface_hash]
            hashes = core_cip_dict .keys()
            random.shuffle(hashes)
            for core_hash in hashes:
                yield core_cip_dict[core_hash]

    def choose_cip(self, graph):
        """
            selects a chip randomly from the graph.
            root is a node_node and not an edge_node
            radius and thickness are chosen to fit the grammars radius and thickness
        """
        node = random.choice(graph.nodes())
        if 'edge' in graph.node[node]:
            node = random.choice(graph.neighbors(node))

        # random radius and thickness
        radius = random.choice(  self.grammar_functions.radius_list  )
        thickness = random.choice(self.grammar_functions.thickness_list)
        return  extract.extract_core_and_interface(node, graph, [radius], [thickness])



# ok moving this here instead of leaving it where it belongs prevents pickling errar ..
#dont quite get it ...        
def improve_loop_multi(x):
    return x[2].improve_loop(x[0], x[1])


################ALL THE THINGS HERE SERVE TO LEARN A GRAMMAR ############

class subgraphdatac:
    def __init__(self):
        self.count = 0


class local_substitutable_graph_grammar:
    """
    i had this class inherit from default dict, but that breaks joblib oOo
    and i cant load anymore.
    """
    # move all the things here that are needed to extract grammar
    def __init__(self, ddict, radius_list, thickness_list, core_interface_pair_remove_threshold=3,
                 interface_remove_threshold=2):
        self.data = ddict
        self.interface_remove_threshold = interface_remove_threshold
        self.radius_list = radius_list
        self.thickness_list = thickness_list
        self.core_interface_pair_remove_threshold = core_interface_pair_remove_threshold
        self.vectorizer = myutils.my_vectorizer()


    def readgraphs(self, graphs, n_jobs=4):
        if n_jobs == 1:
            self.readgraphs_single(graphs)
        else:
            self.readgraphs_multi(graphs, n_jobs)

    def grammar_add_core_interface_data(self, cid):
        if cid.interface_hash not in self.data:
            self.data[cid.interface_hash] = {}

        subgraphdata = subgraphdatac()
        if cid.core_hash in self.data[cid.interface_hash]:
            subgraphdata = self.data[cid.interface_hash][cid.core_hash]
        else:
            self.data[cid.interface_hash][cid.core_hash] = subgraphdata
        subgraphdata.count += 1
        if subgraphdata.count == self.core_interface_pair_remove_threshold:
            subgraphdata.graph = cid.graph
            subgraphdata.radius = cid.radius
            subgraphdata.thickness = cid.thickness

    def readgraphs_single(self, graphs):
        for gr in graphs:
            for core_interface_data_list in extract_cores_and_interfaces(gr, self.radius_list, self.thickness_list,self.vectorizer ):
                for cid in core_interface_data_list:
                    self.grammar_add_core_interface_data(cid)

    def readgraphs_multi(self, graphs, n_jobs):
        problems = itertools.izip(graphs, itertools.repeat(self.radius_list), itertools.repeat(self.thickness_list),itertools.repeat(self.vectorizer) )
        pool = Pool(processes=n_jobs)
        it = pool.imap_unordered(extract_cores_and_interfaces_multi, problems, 10)
        for core_interface_data_listlist in it:
            for core_interface_data_list in core_interface_data_listlist:
                for cid in core_interface_data_list:
                    self.grammar_add_core_interface_data(cid)
                    # rename to trim // pass trimvalue

    def clean(self):
        for interface in self.data.keys():
            for core in self.data[interface].keys():
                if self.data[interface][core].count < self.core_interface_pair_remove_threshold:
                    self.data[interface].pop(core)
            if len(self.data[interface]) < self.interface_remove_threshold:
                self.data.pop(interface)


def extract_cores_and_interfaces(graph, radius_list, thickness_list,vectorizer):
    # expand the graph
    graph = myutils.expand_edges(graph)
    vectorizer._label_preprocessing(graph)
    for node in graph.nodes_iter():
        if 'edge' in graph.node[node]:
            continue
        core_interface_list = extract.extract_core_and_interface(node, graph, radius_list, thickness_list)
        if len(core_interface_list) > 0:
            yield core_interface_list


def extract_cores_and_interfaces_multi(x):
    # expand the graph
    graph, radius_list, thickness_list, vectorizer = x
    graph = myutils.expand_edges(graph)
    vectorizer._label_preprocessing(graph)
    ret = []
    for node in graph.nodes_iter():
        if 'edge' in graph.node[node]:
            continue
        core_interface_list = extract.extract_core_and_interface(node, graph, radius_list, thickness_list)
        if len(core_interface_list) > 0:
            ret.append(core_interface_list)
    return ret


# ############################### FEASIBILITY CHECKER ###################


class feasibility_checker():
    def __init__(self):
        self.checklist = []
        self.checklist.append(defaultcheck)

    def check(self, ng):
        for f in self.checklist:
            if f(ng) == False:
                return False
        return True


def defaultcheck(ng):
    for node_id in ng.node_iter():
        if 'edge' in ng.node[node_id]:
            if len(ng.neighbors(node_id)) != 2:
                return False
    return True



