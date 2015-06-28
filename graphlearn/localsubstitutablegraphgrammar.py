from multiprocessing import Pool, Manager
import graphtools
import dill
from eden import grouper
from eden.graph import Vectorizer
import logging
from coreinterfacepair import CoreInterfacePair
logger = logging.getLogger(__name__)




class LocalSubstitutableGraphGrammar(object):

    """
    the grammar.
        can learn from graphs
        will save all the cips in neat dictionaries
        contains also convenience functions to make it nicely usable
    """
    # move all the things here that are needed to extract grammar

    def __init__(self, radius_list=None, thickness_list=None, core_interface_pair_remove_threshold=3, complexity=3,
                 interface_remove_threshold=2, nbit=20, node_entity_check=lambda x, y: True):
        self.grammar = {}
        self.interface_remove_threshold = interface_remove_threshold
        self.radius_list = radius_list
        self.thickness_list = thickness_list
        self.core_interface_pair_remove_threshold = core_interface_pair_remove_threshold
        self.vectorizer = Vectorizer(complexity=complexity)
        self.hash_bitmask = 2 ** nbit - 1
        self.nbit = nbit
        # checked when extracting grammar. see graphtools
        self.node_entity_check = node_entity_check

    def preprocessing(self, n_jobs=0, same_radius=False, same_core_size=0, probabilistic_core_choice=False):
        '''
            sampler will use this when preparing sampling
        '''
        if self.__dict__.get('locked', False):
            logger.debug(
                'skipping preprocessing of grammar. (we lock the grammar after sampling, so the preprocessing does not rerun every time we graphlearn.sample())')
            return
        else:
            logger.debug('preprocessing grammar')
        if same_radius:
            self._add_same_radius_quicklookup()
        if same_core_size:
            self._add_core_size_quicklookup()
        if probabilistic_core_choice:
            self._add_frequency_quicklookup()
        if n_jobs > 1:
            self._multicore_transform()

        self.locked = True

    def fit(self, G_iterator, n_jobs,batch_size=10):
        self._read(G_iterator, n_jobs,batch_size=batch_size)
        self.clean()

    def _multicore_transform(self):
        '''
        this turns the grammar into a managed dictionary which we need for multiprocessing

        note that we dont do this per default because then we cant save the grammar anymore
        while keeping the manager outside the object
        '''

        # do nothing if transform already happened
        if type(self.grammar) != dict:
            return
        # move the grammar into a manager object...
        manager = Manager()
        shelve = manager.dict()
        for k, v in self.grammar.iteritems():
            md = manager.dict()
            for k2, v2 in v.iteritems():
                md[k2] = v2
            shelve[k] = md
        self.grammar = shelve

    def _revert_multicore_transform(self):
        # only if we are managed we need to do this
        if type(self.grammar) != dict:
            shelve = {}
            for k, v in self.grammar.iteritems():
                md = {}
                for k2, v2 in v.iteritems():
                    md[k2] = v2
                shelve[k] = md
            self.grammar = shelve

    def difference(self, other_grammar, substract_cip_count=False):

        # i assume set() my even be faster than sort...
        # should be even faster to just test -_-

        # mykeys=set(my.keys())
        # otherkeys=set(other_grammar.keys())
        for ihash in self.grammar.keys():
            if ihash in other_grammar:
                # so if my ihash is in both i can compare the corehashes
                for chash in self.grammar[ihash].keys():
                    if chash in other_grammar[ihash]:
                        if substract_cip_count:
                            self.grammar[ihash][chash].count -= self.grammar[ihash][chash].count
                        else:
                            self.grammar[ihash].pop(chash)
        if substract_cip_count:
            self.clean()

    def union(self, other_grammar, cip_count_function=sum):
        '''
        we build the union of grammars..
        '''

        for ihash in other_grammar.keys():
            if ihash in self.grammar:
                # ok so we are in the intersection...
                for chash in other_grammar[ihash]:
                    if chash in self.grammar[ihash]:

                        self.grammar[ihash][chash].counter = cip_count_function(
                            self.grammar[ihash][chash].counter, other_grammar[ihash][chash].counter)
                    else:
                        self.grammar[ihash][chash] = other_grammar[ihash][chash]
            else:
                self.grammar[ihash] = other_grammar[ihash]

    def intersect(self, other_grammar):
        # counts will be added...

        for ihash in self.grammar.keys():
            if ihash in other_grammar:
                # so if self.grammar ihash is in both i can compare the corehashes
                for chash in self.grammar[ihash].keys():
                    if chash in other_grammar[ihash]:
                        self.grammar[ihash][chash].counter = min(self.grammar[ihash][chash].counter,
                                                                 other_grammar[ihash][chash].counter)
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

    def _add_same_radius_quicklookup(self):
        '''
            there is now self.radiuslookup{ interfacehash: {radius:[list of corehashes with that ihash and radius]} }
        '''
        self.radiuslookup = {}
        for interface in self.grammar.keys():
            radius_lookup = [[]] * (max(self.radius_list) + 1)

            for core in self.grammar[interface].keys():
                radius = self.grammar[interface][core].radius
                if radius in radius_lookup:
                    radius_lookup[radius].append(core)
                else:
                    radius_lookup[radius] = [core]
            self.radiuslookup[interface] = radius_lookup

    def _add_core_size_quicklookup(self):
        '''
            there is now self.radiuslookup{ interfacehash: {radius:[list of corehashes with that ihash and radius]} }
        '''
        self.core_size = {}
        for interface in self.grammar.keys():
            core_size = {}
            for core in self.grammar[interface].keys():
                nodes_count = self.grammar[interface][core].core_nodes_count
                if nodes_count in core_size:
                    core_size[nodes_count].append(core)
                else:
                    core_size[nodes_count] = [core]
            self.core_size[interface] = core_size

    def _add_frequency_quicklookup(self):
        '''
            how frequent is a core?
        '''
        self.frequency = {}
        # for every interface
        for interface in self.grammar.keys():
            # we create a dict...
            core_frequency = {}
            #  fill it
            for hash, cip in self.grammar[interface].items():
                core_frequency[hash] = cip.count
            # and attach it to the freq lookup
            self.frequency[interface] = core_frequency

    def _read(self, graphs, n_jobs=-1, batch_size=20):
        '''
        we extract all chips from graphs of a graph iterator
        we use n_jobs processes to do so.
        '''

        # if we should use only one process we use read_single ,  else read_multi
        if n_jobs == 1:
            self._read_single(graphs)
        else:
            self._read_multi(graphs, n_jobs, batch_size)

    def _add_core_interface_data(self, cip):
        '''
            cid is a core interface data instance.
            we will add the cid to our grammar.
        '''

        # initialize gramar[interfacehash] if necessary
        if cip.interface_hash not in self.grammar:
            self.grammar[cip.interface_hash] = {}

        # initialize or get grammar[interfacehash][corehash]
        if cip.core_hash in self.grammar[cip.interface_hash]:
            grammar_cip = self.grammar[cip.interface_hash][cip.core_hash]
        else:
            grammar_cip = CoreInterfacePair()
            self.grammar[cip.interface_hash][cip.core_hash] = grammar_cip


        # put new information in the subgraph_data
        # we only save the count until we know that we will keep the actual cip
        grammar_cip.count += 1
        if grammar_cip.count == self.core_interface_pair_remove_threshold:
            grammar_cip.__dict__.update(cip.__dict__)
            grammar_cip.count = self.core_interface_pair_remove_threshold

    def _read_single(self, graphs):
        """
            for graph in graphs:
                get cips of graph
                    put cips into grammar
        """
        for gr in graphs:
            problem = (
                gr, self.radius_list, self.thickness_list, self.vectorizer, self.hash_bitmask, self.node_entity_check)

            for core_interface_data_list in extract_cores_and_interfaces(problem):
                for cip in core_interface_data_list:
                    self._add_core_interface_data(cip)

    def _read_multi(self, graphs, n_jobs, batch_size):
        """
        will take graphs and to multiprocessing to extract their cips
        and put these cips in the grammar




        multiprocessing takes lots of memory, my theory is, that the myeden.multiprocess
        materializes the iterator too fast
        """

        # generate iterator of problem instances
        '''
        problems = itertools.izip(graphs, itertools.repeat(self.radius_list),
                                  itertools.repeat(self.thickness_list),
                                  itertools.repeat(self.vectorizer),
                                  itertools.repeat(self.hash_bitmask),
                                  itertools.repeat(self.node_entity_check)
                                  )
        '''

        # distributing jobs to workers
        # result = pool.imap_unordered(extract_cores_and_interfaces, problems, 10)

        if n_jobs > 1:
            pool = Pool(processes=n_jobs)
        else:
            pool = Pool()

        # extract_c_and_i = lambda batch,args: [ extract_cores_and_interfaces(  [y]+args ) for y in batch ]

        results = pool.imap_unordered(extract_cips, self._multi_process_argbuilder(graphs, batch_size=batch_size))

        # the resulting chips can now be put intro the grammar
        for batch in results:
            for exci in batch:
                if exci:  # exci might be None because the grouper fills up with empty problems
                    for exci_result_per_node in exci:
                        for cid in exci_result_per_node:
                            self._add_core_interface_data(cid)
        pool.close()
        pool.join()

    def _multi_process_argbuilder(self, graphs, batch_size=10):
        args = [self.radius_list, self.thickness_list, self.vectorizer, self.hash_bitmask, self.node_entity_check]
        function = extract_cores_and_interfaces
        for batch in grouper(graphs, batch_size):
            yield dill.dumps((function, args, batch))


def extract_cips(what):
    f, args, graph_batch = dill.loads(what)
    return [f([y] + args) for y in graph_batch]


def extract_cores_and_interfaces(parameters):
    # happens if batcher fills things up with null
    if parameters[0] is None:
        return None
    try:
        # unpack arguments, expand the graph
        graph, radius_list, thickness_list, vectorizer, hash_bitmask, node_entity_check = parameters
        graph = vectorizer._edge_to_vertex_transform(graph)
        cips = []
        for node in graph.nodes_iter():
            if 'edge' in graph.node[node]:
                continue
            cip_list = graphtools.extract_core_and_interface(node, graph, radius_list, thickness_list,
                                                                        vectorizer=vectorizer,
                                                                        hash_bitmask=hash_bitmask,
                                                                        filter=node_entity_check)
            if cip_list:
                cips.append(cip_list)
        return cips
    except:
        # as far as i remember this should almost never happen,
        # if it does you may have a bigger problem.
        # so i put this in info
        logger.info( "extract_cores_and_interfaces_died" )
        logger.info( parameters )
