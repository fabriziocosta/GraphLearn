import networkx as nx
import collections
import graphlearn3.test.transformutil as util
import graphlearn3.lsgg_cip






class Cycler():

    def _compute_cycle_node_name(self, g, cycle):
        return cycle, hash( tuple(sorted( [g.node[n]['label'] for n in cycle] ))) , hash( tuple(sorted(cycle) ))

    def _merge(self, graph, cycle_name_id):
        cycle, name,idd = cycle_name_id
        # add node
        graph.add_node(idd, label=name,contracted=set(cycle))

        # remove old nodes, but keep their edges
        for i in cycle:
            if i in graph:
                util.merge_edge(graph, idd, i)
                graph.graph['cycdic'][i]=idd
            else:
                # node was already removed -> the node musst be in 2 cycles!
                # since the node was removed before we know that its edges live on in the cycle representing node
                abstr_of_i  = graph.graph['cycdic'][i]
                graph.add_edge(abstr_of_i,idd, label='interabstredge')



    def encode(self, graphs):
        for e in graphs:
            yield self.encode_single(e)

    def encode_single(self,graph):
        # new layer graph
        layer=nx.Graph(graph)

        # set contracted attribute
        for n,d in layer.nodes(data=True):
            d['contracted']=set([n])

        # contract
        layer.graph['cycdic']={}
        for cycle in nx.cycle_basis(graph):
            self._merge(layer, self._compute_cycle_node_name(graph, cycle))

        # result :)
        layer.graph['original']=graph
        return layer

    def decode(self,graphs):
        '''there is nothing to do because the output is already gl.valid_gl_graph :D'''
        return graphs







def test_cycle():
    import graphlearn3.util.util as utilz
    c=Cycler()
    G=utilz.test_get_circular_graph()
    result=c.encode_single(G)
    assert (utilz.valid_gl_graph(result)==True)




