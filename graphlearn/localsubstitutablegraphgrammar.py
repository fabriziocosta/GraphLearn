import utils.myeden as graphlearn_utils
import itertools
from multiprocessing import Pool, Manager
import graphtools


################ALL THE THINGS HERE SERVE TO LEARN A GRAMMAR ############


class coreInterfacePair:

    """
    this is refered to throughout the code as cip
    it contains the cip-graph and several pieces of information about it.
    """

    def __init__(self, ihash=0, chash=0, graph=0, radius=0, thickness=0, core_nodes_count=0, distance_dict={}):
        self.interface_hash = ihash
        self.core_hash = chash
        self.graph = graph
        self.radius = radius
        self.thickness = thickness
        self.core_nodes_count = core_nodes_count
        self.distance_dict = distance_dict


class LocalSubstitutableGraphGrammar:

    """
    the grammar.
        can learn from graphs
        will save all the cips in neat dictionaries
        contains also convenience functions to make it nicely usable
    """
    # move all the things here that are needed to extract grammar

    def __init__(self, radius_list, thickness_list, core_interface_pair_remove_threshold=3,
                 interface_remove_threshold=2, nbit=20, node_entity_check= lambda x,y:True):
        self.grammar = {}
        self.interface_remove_threshold = interface_remove_threshold
        self.radius_list = radius_list
        self.thickness_list = thickness_list
        self.core_interface_pair_remove_threshold = core_interface_pair_remove_threshold
        self.vectorizer = graphlearn_utils.GraphLearnVectorizer()
        self.hash_bitmask = 2 ** nbit - 1
        self.nbit = nbit
        # checked when extracting grammar. see graphtools
        self.node_entity_check= node_entity_check

    def preprocessing(self,n_jobs=0,same_radius=False,same_core_size=0):
        '''
            sampler will use this when preparing sampling
        '''
        if n_jobs > 0:
            self.multicore_transform()
        if same_radius:
            self.add_same_radius_quicklookup()
        if same_core_size:
            self.add_core_size_quicklookup()

    def fit(self,G_iterator,n_jobs):
        self.read(G_iterator, n_jobs)
        self.clean()

    def multicore_transform(self):
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

    def revert_multicore_transform(self):
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
                        self.grammar[ihash][chash].counter = min(self.grammar[ihash][chash].counter, other_grammar[ihash][chash].counter)
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

    def add_same_radius_quicklookup(self):
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

    def add_core_size_quicklookup(self):
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

    def read(self, graphs, n_jobs=-1):
        '''
        we extract all chips from graphs of a graph iterator
        we use n_jobs processes to do so.
        '''

        # if we should use only one process we use read_single ,  else read_multi
        if n_jobs == 1:
            self.read_single(graphs)
        else:
            self.read_multi(graphs, n_jobs)

    def add_core_interface_data(self, cid):
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
            subgraph_data = coreInterfacePair()
            self.grammar[cid.interface_hash][cid.core_hash] = subgraph_data
            subgraph_data.count = 0

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
            problem = (gr, self.radius_list, self.thickness_list, self.vectorizer, self.hash_bitmask,self.node_entity_check)
            for core_interface_data_list in extract_cores_and_interfaces(problem):
                for cid in core_interface_data_list:
                    self.add_core_interface_data(cid)

    def read_multi(self, graphs, n_jobs):
        """
        will take graphs and to multiprocessing to extract their cips
        and put these cips in the grammar
        """

        # generate iterator of problem instances
        problems = itertools.izip(graphs, itertools.repeat(self.radius_list),
                                  itertools.repeat(self.thickness_list),
                                  itertools.repeat(self.vectorizer),
                                  itertools.repeat(self.hash_bitmask),
                                  itertools.repeat(self.node_entity_check)
                                  )


        # distributing jobs to workers
        #result = pool.imap_unordered(extract_cores_and_interfaces, problems, 10)

        extract_c_and_i = lambda x: [ extract_cores_and_interfaces(y) for y in x ]
        result = graphlearn_utils.multiprocess_classic(problems,extract_c_and_i,n_jobs=n_jobs,batch_size=10)

        # the resulting chips can now be put intro the grammar
        for cidlistlist in result:
            for cidlist in cidlistlist:
                for cid in cidlist:
                    self.add_core_interface_data(cid)



def extract_cores_and_interfaces(parameters):
    if parameters == None:
        return None

    # unpack arguments, expand the graph
    graph, radius_list, thickness_list, vectorizer, hash_bitmask ,node_entity_check= parameters
    graph = graphlearn_utils.expand_edges(graph)
    cips = []
    for node in graph.nodes_iter():
        if 'edge' in graph.node[node]:
            continue
        core_interface_list = graphtools.extract_core_and_interface(node, graph, radius_list, thickness_list,
                                                         vectorizer=vectorizer, hash_bitmask=hash_bitmask,
                                                         node_entity_check=node_entity_check)
        if core_interface_list:
            cips.append(core_interface_list)

    return cips

