'''
WE EXTEND THE MINORDECOMPOSER WITH THESE THINGS:
   -- make a rna decomp
        -- option to format output                 OPTION
        -- option to ignore intloops               OPTION
        -- use the shards                          ALWAYS
        -- rm F-nodes before estimator :)          OPTION


THIS IS THE OLD ANNOTATION FOR THIS FILE, PROBABLY MOSTLY OUTDATED:

    we extend the minor decomposer by 2 properties for forgi abstractions:
    Fnodes: when two contracted nodes are next to each other and the
            subgraph of one is dicsonnected
            problems may arise. to fix this F nodes may be introduced
    fcorrect: remove F nodes from backbone when printing



last update: 2017-03-15
'''


import networkx as nx
from graphlearn01.minor.decompose import MinorDecomposer
from graphlearn01.minor.rna import get_sequence
import graphlearn01.minor.rna as rna
from eden import graph as edengraphtools






class RnaDecomposer(MinorDecomposer):


    def __init__(self,graph=None,output_sequence=False,
                      ignore_internal_loops=False,
                      pre_vectorizer_rm_f=False,
                      pre_vectorizer_nested=False,
                        **kwargs):

        self.output_sequence=output_sequence
        self.ignore_internal_loops = ignore_internal_loops
        self.pre_vectorizer_rm_f=pre_vectorizer_rm_f
        self.pre_vectorizer_nested=pre_vectorizer_nested




        super(self.__class__, self).__init__(graph=graph,**kwargs)




    def make_new_decomposer(self, transformout):
        return RnaDecomposer(transformout,

                            node_entity_check=self.node_entity_check,
                            nbit=self.nbit,
                            base_thickness_list=self.some_thickness_list,
                            include_base=self.include_base,

                            output_sequence=self.output_sequence,
                            ignore_internal_loops= self.ignore_internal_loops,
                            pre_vectorizer_rm_f=self.pre_vectorizer_rm_f,
                            pre_vectorizer_nested=self.pre_vectorizer_nested,
                             calc_contracted_edge_nodes=self.calc_contracted_edge_nodes)


    def rooted_core_interface_pairs(self, root,
                                    thickness_list=None,
                                    for_base=False,
                                    radius_list=[],
                                    base_thickness_list=False):

        """
        Parameters
        ----------
        root:
        thickness:
        args:

        Returns
        -------
        """

        ciplist = super(self.__class__, self).rooted_core_interface_pairs(root,
                                        thickness_list=thickness_list,
                                        for_base=for_base,
                                        radius_list=radius_list,
                                        base_thickness_list=base_thickness_list)



        if not for_base:
            # numbering shards if cip graphs not connected
            for cip in ciplist:
                if not nx.is_weakly_connected(cip.graph):
                    comps = [list(node_list) for node_list in nx.weakly_connected_components(cip.graph)]
                    comps.sort()
                    for i, nodes in enumerate(comps):
                        for node in nodes:
                            cip.graph.node[node]['shard'] = i


        return ciplist

    def out(self):

        if self.output_sequence:
             sequence = get_sequence(self.base_graph())
             return ('',sequence.replace("F",""))

        return self.base_graph()





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

        if self.pre_vectorizer_rm_f and self.pre_vectorizer_nested:
            print "not sure if this works.. rnadecomposer pre_vec_graph"


        if self.pre_vectorizer_rm_f == False:
            return super(self.__class__, self).pre_vectorizer_graph(nested=self.pre_vectorizer_nested)


        # else

        backup = self._unaltered_graph.copy()
        g=self._unaltered_graph
        if g:
            if self.pre_vectorizer_rm_f:
                delset = []
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



        #g2=g.graph['original'] eden should preserve the dict..
        self._unaltered_graph =edengraphtools._edge_to_vertex_transform(g)
        #g.graph['original']=g2
        g = super(self.__class__, self).pre_vectorizer_graph(nested=self.pre_vectorizer_nested)



        #import graphlearn.utils.ascii as asc
        #asc.printrow([g],size=60)
        #exit()

        self._unaltered_graph = backup

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