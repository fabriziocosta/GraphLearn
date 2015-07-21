from multiprocessing import Pool, Manager
import graphtools
import dill
from eden import grouper
from eden.graph import Vectorizer
import logging
import traceback
logger = logging.getLogger(__name__)


class LocalSubstitutableGraphGrammar(object):

    """
    the grammar.
        can learn from graphs
        will save all the cips in neat dictionaries
        contains also convenience functions to make it nicely usable
    """
    # move all the things here that are needed to extract grammar

    def __init__(self, radius_list=None, thickness_list=None, min_cip_count=3, complexity=3,
                 min_interface_count=2, nbit=20, node_entity_check=lambda x, y: True):
        self.grammar = {}
        self.min_interface_count = min_interface_count
        self.radius_list = radius_list
        self.thickness_list = thickness_list
        self.min_cip_count = min_cip_count
        self.vectorizer = Vectorizer(complexity=complexity)
        self.hash_bitmask = 2 ** nbit - 1
        self.nbit = nbit
        # checked when extracting grammar. see graphtools
        self.node_entity_check = node_entity_check

    def preprocessing(self, n_jobs=0, same_radius=False, same_core_size=0, probabilistic_core_choice=False):
        """Preprocess need to be done before sampling.

        Args:
            n_jobs: number of jobs to run for the task
            same_radius: creates same radius data structure
            same_core: creates same core data structure
            probabilistic_core_choice: creates probabilistic core data structure
        """
        # FIXME: resolve this bug!
        # I have hardwired the non locking but resolve it in a more appropriate way!
        self.__dict__['locked'] = False
        if self.__dict__.get('locked', False):
            logger.debug(
                'skipping preprocessing of grammar. (we lock the grammar after sampling, so the preprocessing \
                 does not rerun every time we graphlearn.sample())')
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

    def fit(self, graph_iterator, n_jobs, batch_size=10):
        self._read(graph_iterator, n_jobs, batch_size=batch_size)
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
        """difference between grammars"""
        for interface in self.grammar:
            if interface in other_grammar:
                for core in self.grammar[interface]:
                    if core in other_grammar[interface].keys():
                        if substract_cip_count:
                            self.grammar[interface][core].count -= other_grammar[interface][core].count
                        else:
                            self.grammar[interface].pop(core)

        if substract_cip_count:
            self.clean()

    def union(self, other_grammar):
        """union of grammars"""
        for interface in self.grammar:
            if interface in other_grammar:
                for core in self.grammar[interface]:
                    if core in other_grammar[interface]:
                        self.grammar[interface][core].counter = sum(self.grammar[interface][core].counter,
                                                                    other_grammar[interface][core].counter)
                    else:
                        self.grammar[interface][core] = other_grammar[interface][core]
            else:
                self.grammar[interface] = other_grammar[interface]

    def intersect(self, other_grammar):
        """intersection of grammars"""
        for interface in self.grammar.keys():
            if interface in other_grammar:
                for core in self.grammar[interface].keys():
                    if core in other_grammar[interface]:
                        self.grammar[interface][core].counter = min(self.grammar[interface][core].counter,
                                                                    other_grammar[interface][core].counter)
                    else:
                        self.grammar[interface].pop(core)
            else:
                self.grammar.pop(interface)

    def clean(self):
        """remove cips and interfaces not been seen enough during grammar creation"""
        for interface in self.grammar.keys():
            for core in self.grammar[interface].keys():
                if self.grammar[interface][core].count < self.min_cip_count:
                    self.grammar[interface].pop(core)
            if len(self.grammar[interface]) < self.min_interface_count:
                self.grammar.pop(interface)

    def _add_same_radius_quicklookup(self):
        """adds self.radiuslookup{ interface: { radius:[list of cores] } }"""
        self.radiuslookup = {}
        for interface in self.grammar:
            radius_lookup = [[]] * (max(self.radius_list) + 1)
            for core in self.grammar[interface]:
                radius = self.grammar[interface][core].radius
                if radius in radius_lookup:
                    radius_lookup[radius].append(core)
                else:
                    radius_lookup[radius] = [core]
            self.radiuslookup[interface] = radius_lookup

    def _add_core_size_quicklookup(self):
        """"adds self.core_size{ interface: { core_size:[list of cores] } }"""
        self.core_size = {}
        for interface in self.grammar:
            core_size = {}
            for core in self.grammar[interface]:
                nodes_count = self.grammar[interface][core].core_nodes_count
                if nodes_count in core_size:
                    core_size[nodes_count].append(core)
                else:
                    core_size[nodes_count] = [core]
            self.core_size[interface] = core_size

    def _add_frequency_quicklookup(self):
        """adds self.frequency{ interface: { core_frequency:[list of cores] } }"""
        self.frequency = {}
        for interface in self.grammar:
            core_frequency = {}
            for hash, cip in self.grammar[interface].items():
                core_frequency[hash] = cip.count
            self.frequency[interface] = core_frequency

    def _read(self, graphs, n_jobs=-1, batch_size=20):
        """find all posible cips in graph list and add them to the grammar"""
        if n_jobs == 1:
            self._read_single(graphs)
        else:
            self._read_multi(graphs, n_jobs, batch_size)

    def _add_core_interface_data(self, cip):
        """add the cip to the grammar"""
        interface = cip.interface_hash
        core = cip.core_hash

        if interface not in self.grammar:
            self.grammar[interface] = {}

        if core not in self.grammar[interface]:
            self.grammar[interface][core] = cip

        self.grammar[interface][core].count += 1

    def _read_single(self, graphs):
        """
            for graph in graphs:
                get cips of graph
                    put cips into grammar
        """
        args = self._get_args()
        for gr in graphs:
            problem = [gr] + args
            for core_interface_data_list in self.get_cip_extractor()(problem):
                for cip in core_interface_data_list:
                    self._add_core_interface_data(cip)

    def _read_multi(self, graphs, n_jobs, batch_size):
        """
        like read_single but with multiple processes
        """

        if n_jobs > 1:
            pool = Pool(processes=n_jobs)
        else:
            pool = Pool()

        # extract_c_and_i = lambda batch,args: [ extract_cores_and_interfaces(  [y]+args ) for y in batch ]

        results = pool.imap_unordered(extract_cips,
                                      self._multi_process_argbuilder(graphs, batch_size=batch_size))

        # the resulting chips can now be put intro the grammar
        for batch in results:
            for exci in batch:
                if exci:  # exci might be None because the grouper fills up with empty problems
                    for exci_result_per_node in exci:
                        for cip in exci_result_per_node:
                            self._add_core_interface_data(cip)
        pool.close()
        pool.join()

    def _multi_process_argbuilder(self, graphs, batch_size=10):
        args = self._get_args()
        function = self.get_cip_extractor()
        for batch in grouper(graphs, batch_size):
            yield dill.dumps((function, args, batch))

    '''
    these 2 let you easily change the cip extraction process...

    the problem was that you needed to overwrite read_single AND read_multi when you wanted to change the cip
    extractor. :)
    '''

    def _get_args(self):
        return [self.radius_list,
                self.thickness_list,
                self.vectorizer,
                self.hash_bitmask,
                self.node_entity_check]

    def get_cip_extractor(self):
        return extract_cores_and_interfaces


def extract_cips(what):
    '''
    :param what: unpacks and runs jobs that were packed by the _multi_process_argbuilder
    :return:  [extract_cores_and_interfaces(stuff),extract_cores_and_interfaces(stuff), ...]
    '''
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
        for root_node in graph.nodes_iter():
            if 'edge' in graph.node[root_node]:
                continue
            cip_list = graphtools.extract_core_and_interface(root_node=root_node,
                                                             graph=graph,
                                                             radius_list=radius_list,
                                                             thickness_list=thickness_list,
                                                             vectorizer=vectorizer,
                                                             hash_bitmask=hash_bitmask,
                                                             filter=node_entity_check)

            if cip_list:
                cips.append(cip_list)
        return cips

    except Exception:
        logger.debug(traceback.format_exc(10))
        # as far as i remember this should almost never happen,
        # if it does you may have a bigger problem.
        # so i put this in info
        # logger.info( "extract_cores_and_interfaces_died" )
        # logger.info( parameters )


def extract_core_and_interface_single_root(**kwargs):
    return graphtools.extract_core_and_interface(**kwargs)
