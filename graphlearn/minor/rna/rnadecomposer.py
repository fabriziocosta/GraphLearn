'''
we extend the minor decomposer by 2 properties for forgi abstractions:
Fnodes: when two contracted nodes are next to each other and the subgraph of one is dicsonnected
        problems may arise. to fix this F nodes may be introduced
fcorrect: remove F nodes from backbone when printing
'''


import networkx as nx
from graphlearn.minor.decompose import MinorDecomposer
from graphlearn.minor.rna import get_start_and_end_node


class RnaDecomposer(MinorDecomposer):
    # def core_substitution(self, orig_cip_graph, new_cip_graph):
    #    graph=graphtools.core_substitution( self._base_graph, orig_cip_graph ,new_cip_graph )
    #    return self.__class__( graph, self.vectorizer , self.some_thickness_list)



    def __init__(self, data=[], node_entity_check=lambda x, y: True, nbit=20):
        '''
        Parameters
        ----------
        sequence: string
            rna sequence
        structure: string
            dotbracket
        base_graph: raw graph
            base graph
        abstract_graph: graph
            the abstracted graph

        Returns
        -------
        '''

        sequence = data[0]
        structure = data[1]
        base_graph = data [2]
        abstract_graph = data[3]


        self.ignore_inserts = ignore_inserts
        self.some_thickness_list = base_thickness_list
        self.vectorizer = vectorizer
        self._abstract_graph = abstract_graph
        self._base_graph = base_graph
        self.sequence = sequence
        self.structure = structure
        self.include_base = include_base

        # self._base_graph = converter.sequence_dotbracket_to_graph(
        #                                      seq_info=self.sequence, seq_struct=self.structure)
        # self._base_graph = vectorizer._edge_to_vertex_transform(self._base_graph)
        # self._base_graph = expanded_rna_graph_to_digraph(self._base_graph)

        # normaly anything in the core can be replaced,
        # the mod dict is a way arrounf that rule.. it allows to mark special nodes that can only
        # be replaced by something having the same marker.
        # we dont want start and end nodes to disappear, so we mark them :)
        s, e = get_start_and_end_node(self.base_graph())
        self._mod_dict = {s: 696969, e: 123123123}

    def rooted_core_interface_pairs(self, root, thickness=None,  for_base=False,
                                        hash_bitmask=None,
                                      radius_list=[],
                                      thickness_list=None,
                                      node_filter=lambda x, y: True):
        """

        Parameters
        ----------
        root:
        thickness:
        args:

        Returns
        -------

        """

        ciplist = super(self.__class__, self).rooted_core_interface_pairs(root, thickness, for_base=for_base,
                                        hash_bitmask=hash_bitmask,
                                      radius_list=radius_list,
                                      thickness_list=thickness_list,
                                      node_filter=node_filter)



        # numbering shards if cip graphs not connected
        for cip in ciplist:
            if not nx.is_weakly_connected(cip.graph):
                comps = [list(node_list) for node_list in nx.weakly_connected_components(cip.graph)]
                comps.sort()

                for i, nodes in enumerate(comps):

                    for node in nodes:
                        cip.graph.node[node]['shard'] = i

        '''
        solve problem of single-ede-nodes in the core
        this may replace the need for fix_structure thing
        this is a little hard.. may fix later

        it isnt hard if i write this code in merge_core in ubergraphlearn

        for cip in ciplist:
            for n,d in cip.graph.nodes(data=True):
                if 'edge' in d and 'interface' not in d:
                    if 'interface' in cip.graph.node[ cip.graph.successors(n)[0]]:
                        #problem found
        '''

        return ciplist
    '''
    def out(self):
        # copy and  if digraph make graph
        return self.base_graph()
        # sequence=get_sequence(self.base_graph())
        # return ('',sequence.replace("F",""))
    '''
    def pre_vectorizer_graph(self, nested=True, fcorrect=False, base_only=False):
        """

        Parameters
        ----------
        nested: nested minor + base graph
        fcorrect: introduce nodes that aid the forgi abstraction
        base_only: only return the base graph


        Returns
        -------
            nx.graph
        """

        g = nx.disjoint_union(nx.Graph(self._base_graph), self.abstract_graph())
        if base_only:
            g = self.base_graph().copy()
        node_id = len(g)
        delset = []
        if nested:
            for n, d in g.nodes(data=True):
                if 'contracted' in d and 'edge' not in d:
                    for e in d['contracted']:
                        if 'edge' not in g.node[e] and not (g.node[e]['label'] == 'F' and fcorrect):  # think about this

                            # we want an edge from n to e
                            g.add_node(node_id, edge=True, label='e')
                            g.add_edge(n, node_id, nesting=True)
                            g.add_edge(node_id, e, nesting=True)
                            # g.add_edge( n, e, nesting=True)
                            node_id += 1

        if fcorrect:
            for n, d in g.nodes(data=True):
                if d['label'] == "F":
                    down = self.nextnode(g, n)
                    up = self.nextnode(g, n, down_direction=False)
                    delset.append((n, down, up))

            for r, d, u in delset:
                # print g.node[d[0]]
                # we copy the label of the adjacent edge to r
                g.node[r] = g.node[d[0]].copy()
                # print g.node[r]

                # g.node[r]={"label":'-','edge':True}
                # delete neighbors
                g.remove_nodes_from([d[0], u[0]])

                # rewire neighbors of neighbors
                g.add_edge(r, d[1])
                g.add_edge(u[1], r)
                # print r,d,u

        return g

    def nextnode(self, g, n, down_direction=True):
        '''
        goto the nextnext node in a direction
        '''
        if down_direction:
            f = g.successors
        else:
            f = g.predecessors
        next = f(n)[0]
        return next, f(next)[0]