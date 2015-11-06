
from abstract import AbstractWrapper
from collections import defaultdict
import eden
from graphlearn.processing import PreProcessor
import networkx as nx
import graphlearn.utils.draw as draw

class PreProcessor(PreProcessor):

    def __init__(self,base_thickness_list=[2]):
        self.base_thickness_list= base_thickness_list


    def fit(self,inputs):
        return self

    def fit_transform(self,inputs):
        '''

        Parameters
        ----------
        input : many inputs

        Returns
        -------
        graphwrapper iterator
        '''
        self.fit(inputs)
        return self.transform(inputs)

    def re_transform_single(self, graph):
        '''

        Parameters
        ----------
        graphwrapper

        Returns
        -------
        a postprocessed graphwrapper
        '''
        # mabe a copy?
        return MolecularWrapper(graph, self.vectorizer, self.base_thickness_list)

    def transform(self,inputs):
        '''

        Parameters
        ----------
        inputs : list of things

        Returns
        -------
        graphwrapper : iterator
        '''
        return [MolecularWrapper(self.vectorizer._edge_to_vertex_transform(i), self.vectorizer, self.base_thickness_list) for i in inputs]




class MolecularWrapper(AbstractWrapper):

    def abstract_graph(self):
        if self._abstract_graph== None:
            graph=self.vectorizer._revert_edge_to_vertex_transform(self._base_graph)
            abstract_graph = make_abstract(graph)
            self._abstract_graph= self.vectorizer._edge_to_vertex_transform(abstract_graph)

            for n, d in self._abstract_graph.nodes(data=True):
                if 'contracted' not in d:
                    d['contracted'] = set()


            getabstr = {contra: node for node, d in self._abstract_graph.nodes(data=True) for contra in d.get('contracted', [])}

            for n, d in self._base_graph.nodes(data=True):
                if 'edge' in d:
                    # if we have found an edge node...
                    # lets see whos left and right of it:
                    n1, n2 = self._base_graph.neighbors(n)
                    # case1: ok those belong to the same gang so we most likely also belong there.
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



        return self._abstract_graph


"""

class PostProcessor:
    def __init__(self):
        pass

    def fit(self, other):
        self.vectorizer = other.vectorizer

    def postprocess(self, graph):
        return GraphManager(graph, self.vectorizer)



class GraphManager(gt.GraphManager):

    '''
    these are the basis for creating a fitting an ubersampler
    def get_estimateable(self):
    def get_base_graph(self):
    def get_abstract_graph(self):
    '''

    def __init__(self, graph, vectorizer):
        self.base_graph = vectorizer._edge_to_vertex_transform(graph)
        self.abstract_graph = make_abstract(self.base_graph.copy())

        # in the abstract graph , all the edge nodes need to have a contracted attribute.
        # originaly this happens naturally but since we make multiloops into one loop there are some left out
        def setset(graph):
            for n, d in graph.nodes(data=True):
                if 'contracted' not in d:
                    d['contracted'] = set()
        setset(self.abstract_graph)

    def graph(self):
        # returns an expanded, undirected graph
        # that the eden machine learning can compute
        return nx.disjoint_union(self.base_graph, self.abstract_graph)

    def get_base_graph(self):
        return self.base_graph

    def get_abstract_graph(self):
        return self.abstract_graph

"""



'''
here we invent the abstractor function
'''



def make_abstract(graph):
    '''
    make sure this is not expanded
    '''
    # prepare fast hash function
    def fhash(stuff):
        return eden.fast_hash(stuff, 2 ** 20 - 1)


    # all nodes get their cycle calculated
    for n, d in graph.nodes(data=True):
        d['cycle'] = list(node_to_cycle(graph, n))
        d['cycle'].sort()
        #if 'parent'in d:
        #    d.pop('parent')



    # make sure most of the abstract nodes are created.
    abstract_graph = nx.Graph()
    for n, d in graph.nodes(data=True):
        cyclash = fhash(d['cycle'])
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



    #  HERE THE ACTUAL ABSTRACTION BEGINS

    # connect nodes in the abstract graph
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
                            #l = [other, n]
                            #l.sort()
                            #connector = fhash(l)
                            shared_nodes = abstract_graph.node[other]['contracted'] & d['contracted']
                            if len(shared_nodes)==0:
                                label='e'
                            else:
                                labels = [ord(graph.node[sid]['label']) for sid in shared_nodes]
                                labels.sort()
                                share_hash = fhash(labels)
                                label='share:'+str(share_hash)
                            abstract_graph.add_edge(other,n,label=label)
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
                    for e in graph.node[neigh]['parent']:
                        abstract_graph.add_edge(n, e, label='e')
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
