'''
a library for cips.
cips with same interface are grouped together.
=> they are congruent and their cores can be replaced
'''

from multiprocessing import Pool, Manager
import dill
from eden import grouper
import logging
import traceback

logger = logging.getLogger(__name__)




class lsgg_basic(object):
    def __init__(self, radius_list=[0,1], thickness_list=[1,2], min_cip_count=2,min_interface_count=2):
        self.radius_list = [int(2 * r) for r in radius_list]
        self.thickness_list = [int(2 * t) for t in thickness_list]
        self.min_interface_count = min_interface_count
        self.min_cip_count = min_cip_count
        self.productions = {}
    
    def fit(self, graphmanagerlist, n_jobs=4, batch_size=10, reinit_productions=True):
        if reinit_productions:
            self.productions={}

        self.dataset_size = len(graphmanagerlist)

        self._read(graphmanagerlist, n_jobs, batch_size=batch_size)
        self.clean()
        dataset_size, interface_counts, core_counts, cip_counts = self.size()
        logger.debug('#instances: %d  #interfaces: %d   #cores: %d   #core-interface-pairs: %d' %
                     (dataset_size, interface_counts, core_counts, cip_counts))
        return self
        
    
    def clean(self):
        """remove cips and interfaces not been seen enough during grammar creation"""
        for interface in self.productions.keys():
            for core in self.productions[interface].keys():
                if self.productions[interface][core].count < self.min_cip_count:
                    self.productions[interface].pop(core)
            if len(self.productions[interface]) < self.min_interface_count:
                self.productions.pop(interface)
        self.prep_is_outdated = True


    def _read(self, graphs, n_jobs=-1, batch_size=20):
        """find all posible cips in graph list and add them to the grammar"""
        if n_jobs == 1:
            self._read_single(graphs)
        else:
            self._read_multi(graphs, n_jobs, batch_size)
        self.prep_is_outdated = True

    def _add_core_interface_data(self, cip):
        """add the cip to the grammar"""
        interface = cip.interface_hash
        core = cip.core_hash

        if interface not in self.productions:
            self.productions[interface] = {}

        if core not in self.productions[interface]:
            self.productions[interface][core] = cip

        self.productions[interface][core].count += 1

            #print 'cant find '
            #print self.productions.keys()
            #print self.productions[interface].keys()
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
        jobs_done = 0
        for batch in results:
            for exci in batch:
                if exci:  # exci might be None because the grouper fills up with empty problems
                    for exci_result_per_node in exci:
                        for cip in exci_result_per_node:
                            self._add_core_interface_data(cip)
                jobs_done += 1
                if jobs_done == self.multiprocess_jobcount and self.mp_prepared:
                    pool.terminate()
        pool.close()
        pool.join()

    def _multi_process_argbuilder(self, graphs, batch_size=10):

        args = self._get_args()
        function = self.get_cip_extractor()
        self.multiprocess_jobcount = 0
        self.mp_prepared = False
        for batch in grouper(graphs, batch_size):
            self.multiprocess_jobcount += batch_size
            yield dill.dumps((function, args, batch))
        self.mp_prepared = True

    '''
    these 2 let you easily change the cip extraction process...

    the problem was that you needed to overwrite read_single AND read_multi when you wanted to change the cip
    extractor. :)
    '''

    def _get_args(self):
        return [self.radius_list,
                self.thickness_list]

    def get_cip_extractor(self):
        return extract_cores_and_interfaces


class LocalSubstitutableGraphGrammar(lsgg_basic):
    """
    the grammar.
        can learn from graphs
        will save all the cips in neat dictionaries
        contains also convenience functions to make it nicely usable
    """

    # move all the things here that are needed to extract grammar

    def __init__(self, radius_list=[0,1], thickness_list=[1,2], min_cip_count=2,min_interface_count=2):
        self.radius_list = [int(2 * r) for r in radius_list]
        self.thickness_list = [int(2 * t) for t in thickness_list]
        self.min_interface_count = min_interface_count
        self.min_cip_count = min_cip_count
        self.prep_is_outdated = True
        self.productions = {}


    def __str__(self):
        s=''
        for k,v in self.__dict__.items():

            if not isinstance(v ,type({})):
                s+= "%s %s \n" % ( k,str(v) )
            else:
                s+= "%s %s \n" % (k,str(len(v)))
        return s



    def preprocessing(self,
                      n_jobs=0,
                      core_size_required=False,
                      probabilistic_core_choice=False,
                      score_cores=False,
                      score_cores_vectorizer=None,
                      score_cores_estimator=None,
                      bytrial=False):
        """
        Preprocess need to be done before sampling.

        Parameters
        ----------
        n_jobs: int
            how many jobs to use
        core_size_required: int
            pass
        probabilistic_core_choice: bool
            choose cores according to frequency
            creates probabilistic core data structure

        score_cores: bool, False
            creates self.score_core_dict if enabled.
            needs vectorizer and estimator to do the annotation.

        Returns
        -------

        """
        
        logger.log(5,'preprocessing grammar')
        if core_size_required:
            if self.prep_is_outdated or 'core_size' not in self.__dict__:
                self._add_core_size_quicklookup()

        if probabilistic_core_choice:
            if self.prep_is_outdated or 'frequency' not in self.__dict__:
                self._add_frequency_quicklookup()

        if score_cores:
            self.score_core_dict = {}
            for interface in self.productions.keys():
                for core in self.productions[interface].keys():
                    graph = self.productions[interface][core].graph.copy()
                    transformed_graph = score_cores_vectorizer.transform([graph])
                    score = score_cores_estimator.predict(
                        transformed_graph)  # cal_estimator.predict_proba(transformed_graph)[0, 1]
                    self.score_core_dict[core] = score
        if bytrial:
            self.bytrial_normalise_all()

        self.prep_is_outdated = False
        if n_jobs > 1:
            #pass
            self._multicore_transform()

    def bytrial_normalise_all(self):
        for interface_hash,dic in self.productions.items():
            self.bytrial_normalise_dict(dic,True)


    def bytrial_update(self,cip,scorediff):

        interface = self.productions[cip.interface_hash]
        # prints the status ooo
        #print map(lambda k: interface[k].bytrialscore,interface.keys())

        score = interface[cip.core_hash].bytrialscore + scorediff
        interface[cip.core_hash].bytrialscore = max(0,score)
        self.bytrial_normalise_dict(interface)


    def bytrial_normalise_dict(self,dic,usefreq=False):
        if usefreq:
            stuff = float( sum([ dic[core].count for core in dic]))
            for core_hash in dic:
                dic[core_hash].bytrialscore=dic[core_hash].count/stuff
        else:
            stuff = float(sum([ dic[core].bytrialscore for core in dic]))
            for core_hash in dic:
                dic[core_hash].bytrialscore= dic[core_hash].bytrialscore/stuff




    def size(self):
        interface_counts = len(self.productions)
        cip_counts = 0
        core_set = set()
        for interface in self.productions:
            for core in self.productions[interface]:
                core_set.add(core)
            cip_counts += len(self.productions[interface])
        core_counts = len(core_set)
        return self.dataset_size, interface_counts, core_counts, cip_counts

    def _multicore_transform(self):
        '''
        this turns the grammar into a managed dictionary which we need for multiprocessing

        note that we dont do this per default because then we cant save the grammar anymore
        while keeping the manager outside the object
        '''

        # do nothing if transform already happened
        if type(self.productions) != dict:
            return
        # move the grammar into a manager object...
        manager = Manager()
        shelve = manager.dict()
        for k, v in self.productions.iteritems():
            md = manager.dict()
            for k2, v2 in v.iteritems():
                md[k2] = v2
            shelve[k] = md
        self.productions = shelve

    def _revert_multicore_transform(self):
        # only if we are managed we need to do this
        if type(self.productions) != dict:
            shelve = {}
            for k, v in self.productions.iteritems():
                md = {}
                for k2, v2 in v.iteritems():
                    md[k2] = v2
                shelve[k] = md
            self.productions = shelve


    def difference(self, other_grammar, substract_cip_count=False):
        """difference between grammars"""
        for interface in self.productions:
            if interface in other_grammar:
                for core in self.productions[interface]:
                    if core in other_grammar[interface].keys():
                        if substract_cip_count:
                            self.productions[interface][core].count -= other_grammar[interface][core].count
                        else:
                            self.productions[interface].pop(core)

        if substract_cip_count:
            self.clean()
        self.prep_is_outdated = True

    def union(self, other_grammar):
        """union of grammars"""
        for interface in self.productions:
            if interface in other_grammar:
                for core in self.productions[interface]:
                    if core in other_grammar[interface]:
                        self.productions[interface][core].counter = \
                            sum(self.productions[interface][core].counter,
                                other_grammar[interface][core].counter)
                    else:
                        self.productions[interface][core] = other_grammar[interface][core]
            else:
                self.productions[interface] = other_grammar[interface]
        self.prep_is_outdated = True

    def intersect(self, other_grammar):
        """intersection of grammars"""
        for interface in self.productions.keys():
            if interface in other_grammar:
                for core in self.productions[interface].keys():
                    if core in other_grammar[interface]:
                        self.productions[interface][core].counter = \
                            min(self.productions[interface][core].counter,
                                other_grammar[interface][core].counter)
                    else:
                        self.productions[interface].pop(core)
            else:
                self.productions.pop(interface)
        self.prep_is_outdated = True



    def _add_core_size_quicklookup(self):
        """"adds self.core_size{ interface: { core_size:[list of cores] } }"""
        logger.debug('adding core size lookup to lsgg')
        self.core_size = {}
        for interface in self.productions:
            for core in self.productions[interface]:
                self.core_size[core] = self.productions[interface][core].core_nodes_count

        '''
        for interface in self.productions:
            core_size = {}
            for core in self.productions[interface]:
                nodes_count = self.productions[interface][core].core_nodes_count

                if nodes_count in core_size:
                    core_size[nodes_count].append(core)
                else:
                    core_size[nodes_count] = [core]
            self.core_size[interface] = core_size
        '''

    def _add_frequency_quicklookup(self):
        """adds self.frequency{ interface: { core_frequency:[list of cores] } }"""
        self.frequency = {}
        for interface in self.productions:
            core_frequency = {}
            for hash, cip in self.productions[interface].items():
                core_frequency[hash] = cip.count
            self.frequency[interface] = core_frequency

   


'''
these are external  for multiprocessing reasons.
'''


def extract_cips(what):
    """

    Parameters
    ----------
    what: unpacks and runs jobs that were packed by the _multi_process_argbuilder

    Returns
    -------
        [extract_cores_and_interfaces(stuff),extract_cores_and_interfaces(stuff), ...]
    """

    f, args, graph_batch = dill.loads(what)
    return [f([y] + args) for y in graph_batch]


def extract_cores_and_interfaces(parameters):
    # happens if batcher fills things up with null
    if parameters[0] is None:
        return None
    try:
        # unpack arguments, expand the graph
        graphmanager, radius_list, thickness_list = parameters
        return graphmanager.all_core_interface_pairs(radius_list=radius_list,
                                                     thickness_list=thickness_list)

    except Exception:
        logger.debug(traceback.format_exc(10))
        # as far as i remember this should almost never happen,
        # if it does you may have a bigger problem.
        # so i put this in info
        # logger.info( "extract_cores_and_interfaces_died" )
        # logger.info( parameters )

# def extract_core_and_interface_single_root(**kwargs):
#    return graphtools.extract_core_and_interface(**kwargs)
