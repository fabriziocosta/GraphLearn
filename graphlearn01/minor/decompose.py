'''
decomposer for graphs and their minors.
extends the cips a normal decomposer is working with by cips that
take care of the minor graphs.
'''
from eden_extra.modifier.graph import vertex_attributes
from eden_extra.modifier.graph.structure import contraction

import graphlearn01.compose
import graphlearn01.decompose as graphtools
from graphlearn01.decompose import Decomposer
import random
import logging
import networkx as nx
from graphlearn01.utils import draw
import eden.display as edraw
import eden
import traceback
logger = logging.getLogger(__name__)
# from eden.graph import Vectorizer
from eden import graph as edengraphtools
import graphlearn01.utils as utils
import graphlearn01.utils.ascii as ascii



class MinorDecomposer(Decomposer):
    '''
    a wrapper normally wraps a graph.
    here we wrap a graph and also take care of its minor.
    '''

    def compress_layers(self):
        '''
        might be the same as in the cascade, where all the intermediary layers are removed

        .. the cascade one tabun works better..
        '''
        # only compress if there is more than 1 layer to compress
        if self.abstract_graph().graph.get('contracted_layers', 0) > 1:

            # ok when we are done there is only one layer :)
            self.abstract_graph().graph['contracted_layers'] = 1

            # a function to traverse the base graph
            def get_leafes(graph, node):
                if graph.node[node].get('contracted', 0) == 0:
                    return [node]
                else:
                    ret = []
                    for node in graph.node[node]['contracted']:
                        ret += get_leafes(graph.graph['original'], node)
                    return ret

            # compress each node :)
            for n, d in self.abstract_graph().nodes(data=True):
                res = []
                for node in d['contracted']:
                    res += get_leafes(self.base_graph(), node)
                d['contracted'] = res
                d['layer'] = 1

            # set the base_graph
            graph = self.abstract_graph()
            while 'original' in graph.graph:
                graph = graph.graph['original']
            self._base_graph = graph
            self._abstract_graph.graph['original'] = graph
        return self



    def get_layers(self):
        '''the idea is to return the layers as list of grpahs'''
        graph = self._unaltered_graph
        result=[graph]
        while 'original' in graph.graph:
            graph = graph.graph['original']
            result.append(graph)
        return result

    def pre_vectorizer_graph(self, nested=True):
        '''
        generate the graph that will be used for evaluation ( it will be vectorized by eden and then used
        in a machine learning scheme).

        Parameters
        ----------
        nested: bool
            the graph returned here is the union of graph minor and the base graph.
            nested decides wether there edges between nodes in the base graph and their
            representative in the graph minor. these edges have the attribute 'nested'.

        Returns
        -------
            nx.graph


        if nested:
            # before we make the union we need to save the ids of all nodes in the base graph

            for n, d in self._base_graph.nodes(data=True):
                d["ID"] = n
            for n, d in self.abstract_graph().nodes(data=True):
                d.pop("ID",None)
        '''

        # transfer layer information to the nodes (otherwise it will be lost)
        graph = self._unaltered_graph
        while 'original' in graph.graph:
            def f(n, d): d['layer'] = graph.graph.get('layer', 0)

            utils.node_operation(graph, f)
            graph = graph.graph['original']

        # make union of everything
        graph = self._unaltered_graph
        graphs = [graph]
        while 'original' in graph.graph:
            graphs.append(graph.graph['original'])
            graph = graph.graph['original']

        # draw.graphlearn(graphs, vertex_label='id')
        try:
            g = nx.union_all(graphs)
        except:
            print 'decompose prevec graph union failed. '
            print ascii.nx_to_ascii(graph,xmax=30,ymax=15)
            #draw.graphlearn(graphs, vertex_label='id')
            # nobody cares... i just need to fix the overlap in ids
            #import graphlearn.minor.old.rnasampler as egraph
            #graphs = map (egraph._revert_edge_to_vertex_transform, graphs)
            #draw.graphlearn(graphs, vertex_label='id', font_size=7)

        if nested:
            # edge_nodes -> edges
            # then look at the contracted nodes to add dark edges.
            # g  = edengraphtools._revert_edge_to_vertex_transform(g)
            try:
                # updating the contracted sets
                # reconstrdict={  d["ID"]:n  for n,d in g.nodes(data=True) if "ID" in d  }
                # for n, d in g.nodes(data=True):
                #    if 'contracted' in d:
                #        d['contracted']=set( [reconstrdict[e] for e in d['contracted']] )


                for node_union_graph, d in g.nodes(data=True):
                    if 'contracted' in d:
                        for e in d['contracted']:
                            if e in g.nodes() and "edge" not in graph.node.get(e,{}): # graph is the original graph.
                                g.add_edge(node_union_graph, e, nesting=True, label='')
                g=eden.graph._revert_edge_to_vertex_transform(g)
            except:
                print 'can not build nested graph... input looks like this:'
                #draw.graphlearn(self._unaltered_graph.graph['original'], vertex_label='id', size=15)
                #draw.graphlearn(self._unaltered_graph, vertex_label='contracted', size=15)

        # add labels to all edges ( this is needed for eden. .. bu
        # g = fix_graph(g)

        return g

    def abstract_graph(self):
        '''

        Returns
        -------
        nx.graph
            returns the graph minor

        here i calculate the minor on demand.
        it is usualy more convenient to calculate the minor in the proprocessor.
        '''
        if self._abstract_graph == None:
            self._abstract_graph = make_abstract(self._base_graph)
            print 'check if anything went hirribly wrong oO because ' \
                  'make_abstract doesnt look like its making anything abstract'
        return self._abstract_graph




    def _prepare_extraction(self):
        # somehow check that there are only 2 layers
        # oO

        # lets get started
        if '_base_graph' in self.__dict__:
            return

        self._base_graph= edengraphtools._edge_to_vertex_transform(self._unaltered_graph.graph['original'].copy())
        #self._base_graph= edengraphtools._edge_to_vertex_transform(self.get_layers()[-1].copy()) # does not work/// why?
        self._abstract_graph = edengraphtools._edge_to_vertex_transform(self._unaltered_graph.copy() )



        # now i want to add the edges of the base graph to the contracted set of the abstract :)


        # SOME TRNASFORMERS (EG RNA) want to do this themselfes also the code below is only for undirected graphs..
        # sososo
        if not self.calc_contracted_edge_nodes:
            #print 'mindecomp _prep_extraction.. skipping contracted buulding ... this should only be the case with rna'
            return


        def base_graph_neighbors(n):
            if type(self._base_graph) == nx.DiGraph:
                return set(self._base_graph.neighbors(n)+self._base_graph.predecessors(n))
            else:
                return self._base_graph.neighbors(n)


        #  make a dictionary that maps from base_graph_node -> node in contracted graph
        getabstr = {contra: node for node, d in self._abstract_graph.nodes(data=True) for contra in d.get('contracted', [])}
        # so this basically assigns edges in the base_graph to nodes in the abstract graph.
        for n, d in self._base_graph.nodes(data=True):
            if 'edge' in d:
                # if we have found an edge node...
                # lets see whos left and right of it:
                n1, n2 = base_graph_neighbors(n)
                # case1: ok those belong to the same gang so we most likely also belong there.

                try:
                    if getabstr[n1] == getabstr[n2]:
                        self._abstract_graph.node[getabstr[n1]]['contracted'].add(n)



                    # case2: neighbors belong to different gangs...
                    else:
                        blub = set(self._abstract_graph.neighbors(getabstr[n1])) & set(self._abstract_graph.neighbors(getabstr[n2]))
                        for blob in blub:
                            if 'contracted' in self._abstract_graph.node[blob]:
                                self._abstract_graph.node[blob]['contracted'].add(n)
                            else:
                                self._abstract_graph.node[blob]['contracted'] = set([n])

                except:
                    traceback.print_stack()
                    print "key error in minor decompose.py %d %d %d" % (n1,n2,n)
                    import structout as so
                    def pg(g):
                        for n,d in g.nodes(data=True):
                            d['id']=str(n)
                        return g
                    so.gprint(pg(self._base_graph), size=35, label='id')
                    so.gprint(pg(self._abstract_graph),size=35, label='id')
                    for n,d in self._abstract_graph.nodes(data=True):
                        print d
                    exit()




    def __init__(self, graph=None,
                 node_entity_check=lambda x, y: True,
                 nbit=20, base_thickness_list=[2],
                 calc_contracted_edge_nodes=True,

                 include_base=False):
        '''
        Parameters
        ----------
        calc_contracted_edge_nodes:
            we get an abstract and a base graph from the transformer.
            if calc conc edge node is true we look at the edges of the edge-to-vertex
            transformed abstract graph. and find the corresponding nodes in the base_graphs to write them
            in the contracted set.

        graph: nx.graph
            the graph is the minor graph. it has a .graph['original'] set.
        node_entity_check
        nbit
        base_thickness_list
        include_base
        '''

        # print "asd",data
        self.some_thickness_list = base_thickness_list
        self.calc_contracted_edge_nodes= calc_contracted_edge_nodes
        if graph:
            self._unaltered_graph=graph
            try:
                self._prepare_extraction()
            except Exception as exc:
                print (exc)
                print (traceback.format_exc(10))
                print 'minor decompose minordecomposerInit fail'
                traceback.print_stack()
                exit()
                #'if there is a list instead of a graph, transformerparam num_classes is your friend'
                #for e,d in self._base_graph.nodes(data=True):
                #    print d
                #for e,d in self._abstract_graph.nodes(data=True):
                #    print d

            #self._base_graph = graph.graph['original'].copy()
            #if len(self._base_graph) > 0:
            #    self._base_graph = edengraphtools._edge_to_vertex_transform(self._base_graph)
            #self._abstract_graph = graph
            # self._abstract_graph.graph.pop('original')
            self._mod_dict = self._abstract_graph.graph.get("mod_dict", {})  # this is the default.


        self.include_base = include_base  # enables this: random_core_interface_pair_base, and if asked for all cips, basecips will be there too
        self.node_entity_check = node_entity_check
        self.hash_bitmask = 2 ** nbit - 1
        self.nbit = nbit




    def make_new_decomposer(self, transformout):
        return MinorDecomposer(transformout, node_entity_check=self.node_entity_check,
                               nbit=self.nbit, base_thickness_list=self.some_thickness_list,
                               include_base=self.include_base, calc_contracted_edge_nodes=self.calc_contracted_edge_nodes)  # node_entity_check=self.node_entity_check, nbit=self.nbit)

    def rooted_core_interface_pairs(self, root,
                                    thickness_list=None,
                                    for_base=False,
                                    radius_list=[],
                                    base_thickness_list=False):
        '''
             get cips for a root
        Parameters
        ----------
        root: int
            vertex id

        thickness: list

        for_base:bool
            do we want to extract from the base graph?
            this will produce a normal graphlearn cip without abstract/'coarsening schemes

        **args: dict
            everything needed by extract_cips

        Returns
        -------


        '''
        if base_thickness_list:
            thickness = base_thickness_list
        else:
            thickness = self.some_thickness_list
        if for_base == False:
            return extract_cips(root, self, base_thickness_list=thickness, mod_dict=self._mod_dict,
                                hash_bitmask=self.hash_bitmask,
                                radius_list=radius_list,
                                thickness_list=thickness_list,
                                node_filter=self.node_entity_check)
        else:
            return extract_cips_base(root, self, base_thickness_list=thickness, mod_dict=self._mod_dict,
                                     hash_bitmask=self.hash_bitmask,
                                     radius_list=radius_list,
                                     thickness_list=thickness_list,
                                     node_filter=self.node_entity_check)

    def all_core_interface_pairs(self,
                                 for_base=False,
                                 radius_list=[],
                                 thickness_list=None,
                                 ):
        '''

        Parameters
        ----------
        args

        Returns
        -------

        '''
        graph = self.abstract_graph()
        nodes = filter(lambda x: self.node_entity_check(graph, x), graph.nodes())
        nodes = filter(lambda x: graph.node[x].get('APPROVEDABSTRACTNODE', True), nodes)

        cips = []
        for root_node in nodes:
            if 'edge' in graph.node[root_node]:
                continue
            cip_list = self.rooted_core_interface_pairs(root_node,
                                                        for_base=for_base,
                                                        radius_list=radius_list,
                                                        thickness_list=thickness_list)
            if cip_list:
                cips.append(cip_list)

        if self.include_base:
            graph = self.base_graph()
            for root_node in graph.nodes_iter():
                if 'edge' in graph.node[root_node]:
                    continue
                cip_list = self.rooted_core_interface_pairs(root_node,
                                                            for_base=self.include_base,

                                                            radius_list=radius_list,
                                                            thickness_list=thickness_list)
                if cip_list:
                    cips.append(cip_list)

        return cips

    def random_core_interface_pair(self,
                                   radius_list=None,
                                   thickness_list=None):
        '''
        get a random cip  rooted in the minor
        Parameters
        ----------
        radius_list: list
        thickness_list: list
        **args: dict
            args for rooted_core_interface_pairs

        Returns
        -------
            cip
        '''
        nodes = filter(lambda x: self.node_entity_check(self.abstract_graph(), x), self.abstract_graph().nodes())
        nodes = filter(lambda x: self.abstract_graph().node[x].get('APPROVEDABSTRACTNODE', True), nodes)
        node = random.choice(nodes)
        if 'edge' in self._abstract_graph.node[node]:
            node = random.choice(self._abstract_graph.neighbors(node))
            # random radius and thickness
        radius_list = [random.choice(radius_list)]
        thickness_list = [random.choice(thickness_list)]
        random_something = [random.choice(self.some_thickness_list)]
        return self.rooted_core_interface_pairs(node, base_thickness_list=random_something,
                                                for_base=False,
                                                radius_list=radius_list,
                                                thickness_list=thickness_list)

    def random_core_interface_pair_base(self, radius_list=None, thickness_list=None, hash_bitmask=None,
                                        node_filter=lambda x, y: True):
        '''
        get a random cip, rooted in the base graph
        Parameters
        ----------
        radius_list
        thickness_list
        args

        Returns
        -------

        '''
        if self.include_base == False:
            raise Exception("impossible oOoo")
        node = random.choice(self.base_graph().nodes())
        if 'edge' in self._base_graph.node[node]:
            node = random.choice(self._base_graph.neighbors(node))
            # random radius and thickness
        radius_list = [random.choice(radius_list)]
        thickness_list = [random.choice(thickness_list)]
        random_something = [random.choice(self.some_thickness_list)]
        return self.rooted_core_interface_pairs(node, base_thickness_list=random_something, for_base=True,
                                                radius_list=radius_list,
                                                thickness_list=thickness_list,
                                                )


def check_and_draw(base_graph, abstr):
    '''

    Parameters
    ----------
    base_graph: a base graph
    abstr:  an abstract graph

    Returns
    -------
        check if EVERY node in base_graph is in any abstr.graph.node['contracted']
    '''
    nodeset = set([a for n, d in abstr.nodes(data=True) for a in d['contracted']])
    broken = []
    for n in base_graph.nodes():
        if n not in nodeset:
            broken.append(n)
            base_graph.node[n]['colo'] = .5
    if len(broken) > 0:
        print "FOUND SOMETHING BROKEN:"
        draw.set_ids(base_graph)
        base_graph.graph['info'] = 'failed to see these:%s' % str(broken)
        edraw.draw_graph(base_graph, vertex_label='id', vertex_color='colo', edge_label=None, size=20)
        for e, d in abstr.nodes(data=True):
            d['label'] = str(d.get('contracted', ''))
        edraw.draw_graph(abstr, vertex_label='label', vertex_color=None, edge_label=None, size=20)
        return False
    return True


def make_abstract(graph):
    '''
    graph should be the same expanded graph that we will feed to extract_cips later...
    Parameters
    ----------
    graph


    Returns
    -------

    '''
    if isinstance(graph, nx.DiGraph):
        graph = graph.to_undirected()

    graph2 = edengraphtools._revert_edge_to_vertex_transform(graph)
    graph2 = edge_type_in_radius_abstraction(graph2)
    graph2 = edengraphtools._edge_to_vertex_transform(graph2)

    # find out to which abstract node the edges belong
    # finding out where the edge-nodes belong, because the contractor cant possibly do this
    getabstr = {contra: node for node, d in graph2.nodes(data=True) for contra in d.get('contracted', [])}

    for n, d in graph.nodes(data=True):
        if 'edge' in d:
            # if we have found an edge node...
            # lets see whos left and right of it:
            n1, n2 = graph.neighbors(n)
            # case1: ok those belong to the same gang so we most likely also belong there.
            if getabstr[n1] == getabstr[n2]:
                graph2.node[getabstr[n1]]['contracted'].add(n)

            # case2: neighbors belong to different gangs...
            else:
                blub = set(graph2.neighbors(getabstr[n1])) & set(graph2.neighbors(getabstr[n2]))
                for blob in blub:
                    if 'contracted' in graph2.node[blob]:
                        graph2.node[blob]['contracted'].add(n)
                    else:
                        graph2.node[blob]['contracted'] = set([n])
    return graph2


def edge_type_in_radius_abstraction(graph):
    '''
    feature was removed from the eden library as far as i can tell
    future me: this comment does not make sense, probably i just contract according to surrounding edge labels.


    # the function needs to set a 'contracted' attribute to each node with a set of vertices that
    # are contracted.
    Parameters
    ----------
    graph: any graph   .. what kind? expanded? which flags musst be set?

    Returns
    -------
    an abstract graph with node annotations that refer to the node ids it is contracting
    '''
    # annotate in node attribute 'type' the incident edges' labels
    labeled_graph = vertex_attributes.incident_edge_label(
        [graph], level=2, output_attribute='type', separator='.').next()

    # do contraction
    contracted_graph = contraction(
        [labeled_graph], contraction_attribute='type', modifiers=[], nesting=False).next()
    return contracted_graph


def extract_cips(node,
                 decomposerinstance,
                 base_thickness_list=None,
                 hash_bitmask=None,
                 mod_dict={},
                 radius_list=[],
                 thickness_list=None,
                 node_filter=lambda x, y: True
                 ):
    '''

    Parameters
    ----------
    node: node in the abstract graph
    decomposerinstance
    base_thickness_list
    hash_bitmask
    mod_dict
    argz

    Returns
    -------
        a  list of cips
    '''

    # if not filter(abstract_graph, node):
    #    return []

    # PREPARE
    abstract_graph = decomposerinstance.abstract_graph()
    base_graph = decomposerinstance.base_graph()
    if 'hlabel' not in abstract_graph.node[abstract_graph.nodes()[0]]:
        edengraphtools._label_preprocessing(abstract_graph)
    if 'hlabel' not in base_graph.node[base_graph.nodes()[0]]:
        edengraphtools._label_preprocessing(base_graph)


    # EXTRACT CIPS NORMALY ON ABSTRACT GRAPH
    abstract_cips = graphtools.extract_core_and_interface(node, abstract_graph, radius_list=radius_list,
                                                          thickness_list=thickness_list, hash_bitmask=hash_bitmask,
                                                          node_filter=node_filter)

    # VOR EVERY ABSTRACT CIP: MERGE CORE IN BASE GRAPH AND APPLY CIP EXTRACTON
    cips = []
    for abstract_cip in abstract_cips:
        base_copy, mergeids = merge_core(base_graph.copy(), abstract_graph, abstract_cip)
        base_level_cips = graphtools.extract_core_and_interface(mergeids[0], base_copy, radius_list=[0],
                                                                thickness_list=base_thickness_list,
                                                                hash_bitmask=hash_bitmask, node_filter=node_filter)

        # VOR EVERY BASE CIP: RESTORE CORE  AND  MERGE INFORMATION WITH ABSTRACT CIP
        core_hash = graphtools.graph_hash(base_graph.subgraph(mergeids), hash_bitmask=hash_bitmask)
        abstract_cip.core_nodes_count = len(mergeids)
        for base_cip in base_level_cips:
            cips.append(
                enhance_base_cip(base_cip, abstract_cip, mergeids, base_graph, hash_bitmask, mod_dict, core_hash))

    return cips


def enhance_base_cip(base_cip, abstract_cip, mergeids, base_graph, hash_bitmask, mod_dict, core_hash):
    '''

    Parameters
    ----------
    base_cip: cip
        a cip that was extracted from the base graph
    abstract_cip: cip
        a cip that was extracted from the abstract graph
    mergeids: list of int
        nodes in the base cip that are in the core of the abstract cip
    base_graph: graph
        the base graph
    hash_bitmask: int
        n/c
    mod_dict: dict
        {id in base_graph: modification to interface hash}
        if there is an exceptionaly important nodetype in thebase graph it makes sure
        that every substitution will preserve this nodetype Oo
        used eg to mark the beginning/end of rna sequences.
        endnode can only be replaced by endnode :)
    core_hash:
        hash for the core that will be used in the finished CIP

    Returns
    -------
        a finished? CIP
    '''
    # we cheated a little with the core, so we need to undo our cheating
    whatever = base_cip.graph.copy()
    base_cip.graph = base_graph.subgraph(base_cip.graph.nodes() + mergeids).copy()

    for n in mergeids:
        base_cip.graph.node[n]['core'] = True

    for n, d in base_cip.graph.nodes(data=True):
        if 'core' not in d:
            d['interface'] = True
            d['distance_dependent_label'] = whatever.node[n]['distance_dependent_label']

    base_cip.core_hash = core_hash
    # merging cip info with the abstract graph
    base_cip.interface_hash = eden.fast_hash_4(base_cip.interface_hash,
                                               abstract_cip.interface_hash,
                                               get_mods(mod_dict, mergeids), 0,
                                               hash_bitmask)

    base_cip.core_nodes_count = abstract_cip.core_nodes_count
    base_cip.radius = abstract_cip.radius
    base_cip.abstract_thickness = abstract_cip.thickness

    # i want to see what they look like :)
    base_cip.abstract_view = abstract_cip.graph
    base_cip.distance_dict = abstract_cip.distance_dict
    return base_cip


def merge_core(base_graph, abstract_graph, abstract_cip):
    """
    Parameters
    ----------
    base_graph: base graph. will be consumed
    abstract_graph:  we want the contracted info.. maybe we also find this in the cip.. not sure
    abstract_cip: the abstract cip

    NOTE: the edges in the abstract graph need to point to edges in the basegraph


    Returns
    -------
        we merge all the nodes in the base_graph, that belong to the core of the abstract_cip

    """

    try:
        mergeids = [base_graph_id
                        for radius in range(abstract_cip.radius + 1)
                            for abstract_node_id in abstract_cip.distance_dict.get(radius)
                                for base_graph_id in abstract_graph.node[abstract_node_id]['contracted']]
    except:
        print 'merge core decomp draws a graph'
        draw.graphlearn([base_graph,abstract_graph,abstract_cip.graph],size= 15, contract=False, vertex_label='id', secondary_vertex_label='contracted')
        #draw.graphlearn([base_graph,abstract_graph,abstract_cip.graph],size= 10, contract=True, vertex_label='id', secondary_vertex_label='contracted')
        #draw.graphlearn_layered2(abstract_graph)
        exit()

    # remove duplicates:
    mergeids = list(set(mergeids))

    for node_id in mergeids[1:]:
        graphlearn01.compose.merge(base_graph, mergeids[0], node_id)

    return base_graph, mergeids


'''
a mod_dict is a modification dictionary.
use get_mod_dict to make a dict of nodenumber:associated_hash
if the nodenumber is in the core, the hash gets added to the interfacehash.
'''


def get_mods(mod_dict, nodes):
    su = 0
    for n in nodes:
        if n in mod_dict:
            su += mod_dict[n]
    return su


# here we create the mod dict once we have a graph..

def get_mod_dict(graph):
    return {}


def extract_cips_base(node,
                      graphmanager,
                      base_thickness_list=None,
                      hash_bitmask=None,
                      mod_dict={},
                      radius_list=[],
                      thickness_list=None,
                      node_filter=lambda x, y: True):
    '''
    Parameters
    ----------
    node: int
        id of a node
    graphmanager: graph-wrapper
        the wrapper that contains the graph
    base_thickness_list: [int]
        thickness of SOMETHING
    hash_bitmask: int
        see above
    mod_dict: dict
        see above
    **argz: dict
        more args
        I guess these are meant:
        radius_list=None,
        thickness_list=None,

        node_filter=lambda x, y: True):

    Returns
    -------
        [CIP]
        a list of core_interface_pairs
    '''

    # if not filter(abstract_graph, node):
    #    return []

    # PREPARE
    abstract_graph = graphmanager.abstract_graph()
    base_graph = graphmanager.base_graph()

    if 'hlabel' not in abstract_graph.node[abstract_graph.nodes()[0]]:
        edengraphtools._label_preprocessing(abstract_graph)
    if 'hlabel' not in base_graph.node[base_graph.nodes()[0]]:
        edengraphtools._label_preprocessing(base_graph)

    # LOOK UP ABSTRACT GRAPHS NODE AND
    # EXTRACT CIPS NORMALY ON ABSTRACT GRAPH
    for n, d in abstract_graph.nodes(data=True):
        if node in d['contracted']:
            abs_node = n
            break
    else:
        raise Exception("IMPOSSIBLE NODE")

    abstract_cips = graphtools.extract_core_and_interface(root_node=abs_node, graph=abstract_graph, radius_list=[0],
                                                          thickness_list=thickness_list, hash_bitmask=hash_bitmask,
                                                          node_filter=node_filter)

    # VOR EVERY ABSTRACT CIP: EXTRACT BASE CIP
    cips = []

    for abstract_cip in abstract_cips:

        base_level_cips = graphtools.extract_core_and_interface(node, base_graph, radius_list=radius_list,
                                                                thickness_list=base_thickness_list,
                                                                hash_bitmask=hash_bitmask)
        # VOR EVERY BASE CIP: hash interfaces and save the abstract view
        for base_cip in base_level_cips:
            cores = [n for n, d in base_cip.graph.nodes(data=True) if 'interface' not in d]
            base_cip.interface_hash = eden.fast_hash_4(base_cip.interface_hash,
                                                       abstract_cip.interface_hash,
                                                       get_mods(mod_dict, cores), 1337,
                                                       hash_bitmask)
            base_cip.abstract_view = abstract_cip.graph
            cips.append(base_cip)

    return cips
