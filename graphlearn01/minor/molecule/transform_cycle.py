'''
graphtransformer to generate a minorgraph based on the cycles found.

transoform: graph -> (graph,abstract_graph)
'''


from collections import defaultdict
import eden
from graphlearn01.transform import GraphTransformer
import networkx as nx
import graphlearn01.utils.draw as draw
from eden.graph import Vectorizer
from eden import graph as edengraphtools
import hashlib

class GraphTransformerCircles(GraphTransformer):
    def __init__(self):
        """
        Parameters
        ----------
        base_thickness_list: [int]
                list of thicknesses for the base graph.   radius_list and thickness_list for abstract graph are
                those that are handled by graphlearn

        Returns
        -------
        void
        """
        self.debug=False
        pass

    def wrap(self, graph):
        """

        Parameters
        ----------
        graph: nx.graph

        Returns
        -------
        graphdecomposer
        """

        graph = edengraphtools._edge_to_vertex_transform(graph)
        return (graph,self.abstract(graph))
        #return MinorDecomposer(graph, vectorizer=self.vectorizer, base_thickness_list=self.base_thickness_list,
        #  abstract_graph=self.abstract(graph))


    def transform(self, inputs):
        return   map( self.transform_single, inputs)

    '''
    def transform_single(self, graph):
        res = make_abstract(graph.copy())
        res.graph['original']=graph

        if self.debug:
            draw.graphlearn([res, res.graph['original']], vertex_label='id', secondary_vertex_label='contracted')
            for n,d in res.nodes(data=True):
                print n,d
            draw.graphlearn_layered2([res])
            #self.debug=False

        return res

    '''
    def transform_single(self, graph):
        product = self.abstract(graph)
        product.graph['original']= edengraphtools._edge_to_vertex_transform(graph)
        return product

    def abstract(self, graph):
        graph = edengraphtools._edge_to_vertex_transform(graph)
        tmpgraph = edengraphtools._revert_edge_to_vertex_transform(graph)
        abstract_graph = make_abstract(tmpgraph)
        _abstract_graph = edengraphtools._edge_to_vertex_transform(abstract_graph)

        for n, d in _abstract_graph.nodes(data=True):
            if 'contracted' not in d:
                d['contracted'] = set()

        getabstr = {contra: node for node, d in _abstract_graph.nodes(data=True) for contra in d.get('contracted', [])}

        for n, d in graph.nodes(data=True):
            if 'edge' in d:
                # if we have found an edge node...
                # lets see whos left and right of it:
                n1, n2 = graph.neighbors(n)
                # case1: ok those belong to the same gang so we most likely also belong there.
                if getabstr[n1] == getabstr[n2]:
                    _abstract_graph.node[getabstr[n1]]['contracted'].add(n)

                # case2: neighbors belong to different gangs...
                else:
                    blub = set(_abstract_graph.neighbors(getabstr[n1])) & set(_abstract_graph.neighbors(getabstr[n2]))
                    for blob in blub:
                        if 'contracted' in _abstract_graph.node[blob]:
                            _abstract_graph.node[blob]['contracted'].add(n)
                        else:
                            _abstract_graph.node[blob]['contracted'] = set([n])

        return _abstract_graph

'''
here we invent the abstractor function
'''

def make_abstract(graph):
    '''
    Args:
        graph: unexpanded graph

    Returns:  nx.graph
        the abstraction

    '''
    # prepare fast hash function
    def fhash(stuff):
        return int(hashlib.sha224(str(stuff)).hexdigest(),16)
        #return eden.fast_hash(stuff, 2 ** 20 - 1)



    # all nodes get their cycle calculated
    for n, d in graph.nodes(data=True):
        d['cycle'] = list(node_to_cycle(graph, n))
        d['cycle'].sort()
        d.pop('parent',None)
        # if 'parent'in d:
        #    d.pop('parent')

    # make sure most of the abstract nodes are created.
    abstract_graph = nx.Graph()
    for n, d in graph.nodes(data=True):
        cyclash = fhash(d['cycle'])+ max(graph.nodes())+1
        if cyclash not in abstract_graph.node:
            abstract_graph.add_node(cyclash)
            abstract_graph.node[cyclash]['contracted'] = set(d['cycle'])
            abstract_graph.node[cyclash]['node'] = True
            # it is possible that a node belongs to more than 1 cycle, so...
            # each node gets parents
            for e in d['cycle']:
                node = graph.node[e]
                if 'parent' not in node:
                    node['parent'] = set()
                node['parent'].add(cyclash)

    get_element = lambda x: list(x)[0]

    # FOR ALL ABSTRACT NODES
    for n, d in abstract_graph.nodes(data=True):
        # FIND A LABEL
        if len(d['contracted']) > 1:
            labels = [ord(graph.node[childid]['label']) for childid in d['contracted']]
            labels.sort()
            d['label'] = "cycle%d" % fhash(labels)

        else:
            d['label'] = graph.node[get_element(d['contracted'])]['label']

        # THEN LOOK AT ALL CONTRACTED NODES TO FIND OUT WHAT CONNECTION WE HAVE TO OUR NEIGHBORS
        for base_node in d['contracted']:
            base_neighbors = graph.neighbors(base_node)
            # for all the neighbors
            for neigh in base_neighbors:
                # find out if we have to build a connector node
                if len(graph.node[neigh]['cycle']) > 1 and len(d['contracted']) > 1:

                    for other in graph.node[neigh]['parent']:
                        if other != n:
                            # l = [other, n]
                            # l.sort()
                            # connector = fhash(l)
                            shared_nodes = abstract_graph.node[other]['contracted'] & d['contracted']
                            if len(shared_nodes) == 0:
                                label = 'e'
                            else:
                                labels = [ord(graph.node[sid]['label']) for sid in shared_nodes]
                                labels.sort()
                                share_hash = fhash(labels)
                                label = 'share:' + str(share_hash)
                            abstract_graph.add_edge(other, n, label=label)
                            '''
                            if connector not in abstract_graph.node:
                                # we need to consider making the edge the actual intersect of the two...

                                abstract_graph.add_node(connector)
                                abstract_graph.node[connector]['edge'] = True

                                # abstract_graph.node[connector]['label']='edge'
                                shared_nodes = abstract_graph.node[other]['contracted'] & d['contracted']

                                labels = [ord(graph.node[sid]['label']) for sid in shared_nodes]
                                labels.sort()
                                share_hash = fhash(labels)


                                abstract_graph.node[connector]['label'] = "shared" + str(share_hash)

                                abstract_graph.add_edge(other, connector)
                                abstract_graph.add_edge(connector, n)
                            '''
                else:
                    try:
                        for e in graph.node[neigh]['parent']:
                            abstract_graph.add_edge(n, e, label='e')
                    except:
                        print neigh, e ,n
                        import pprint
                        for n,d in graph.nodes_iter(data=True):
                            pprint.pprint(d)
                            print fhash(d['cycle'])+max(graph.nodes())+1
                        draw.graphlearn([abstract_graph, graph], size=20, vertex_label="label", secondary_vertex_label="parent")
                        exit()
    return abstract_graph


def node_to_cycle(graph, n, min_cycle_size=3):
    """
    :param graph:
    :param n: start node
    :param min_cycle_size:
    :return:  a cycle the node belongs to

    so we start in node n,
    then we expand 1 node further in each step.
    if we meet a node we had before we found a cycle.

    there are 3 possible cases.
        - frontier hits frontier -> cycle of even length
        - frontier hits visited nodes -> cycle of uneven length
        - it is also possible that the newly found cycle doesnt contain our start node. so we check for that
    """

    def close_cycle(collisions, parent, root):
        '''
            we found a cycle, but that does not say that the root node is part of that cycle..S
        '''

        def extend_path_to_root(work_list, parent_dict, root):
            """
            :param work_list: list with start node
            :param parent_dict: the tree like dictionary that contains each nodes parent(s)
            :param root: root node. probably we dont really need this since the root node is the orphan
            :return: cyclenodes or none

             --- mm we dont care if we get the shortest path.. that is true for cycle checking.. but may be a problem in
             --- cycle finding.. dousurururururu?
            """
            current = work_list[-1]
            while current != root:
                work_list.append(parent_dict[current][0])
                current = work_list[-1]
            return work_list[:-1]

        # any should be fine. e is closing a cycle,
        # note: e might contain more than one hit but we dont care
        e = collisions.pop()
        # print 'r',e
        # we closed a cycle on e so e has 2 parents...
        li = parent[e]
        a = [li[0]]
        b = [li[1]]
        # print 'pre',a,b
        # get the path until the root node
        a = extend_path_to_root(a, parent, root)
        b = extend_path_to_root(b, parent, root)
        # print 'comp',a,b
        # of the paths to the root node dont overlap, the root node musst be in the loop
        a = set(a)
        b = set(b)
        intersect = a & b
        if len(intersect) == 0:
            paths = a | b
            paths.add(e)
            paths.add(root)
            return paths
        return False

    # START OF ACTUAL FUNCTION
    no_cycle_default = set([n])
    frontier = set([n])
    step = 0
    visited = set()
    parent = defaultdict(list)

    while frontier:
        # print frontier
        step += 1

        # give me new nodes:
        next = []
        for front_node in frontier:
            new = set(graph.neighbors(front_node)) - visited
            next.append(new)
            for e in new:
                parent[e].append(front_node)

        # we merge the new nodes.   if 2 sets collide, we found a cycle of even length
        while len(next) > 1:
            # merge
            s1 = next[1]
            s2 = next[0]
            merge = s1 | s2

            # check if we havee a cycle   => s1,s2 overlap
            if len(merge) < len(s1) + len(s2):
                col = s1 & s2
                cycle = close_cycle(col, parent, n)
                if cycle:
                    if step * 2 > min_cycle_size:
                        return cycle
                    return no_cycle_default

            # delete from list
            next[0] = merge
            del next[1]
        next = next[0]

        # now we need to check for cycles of uneven length => the new nodes hit the old frontier
        if len(next & frontier) > 0:
            col = next & frontier
            cycle = close_cycle(col, parent, n)
            if cycle:
                if step * 2 - 1 > min_cycle_size:
                    return cycle
                return no_cycle_default

        # we know that the current frontier didntclose cycles so we dont need to look at them again
        visited = visited | frontier
        frontier = next
    return no_cycle_default
