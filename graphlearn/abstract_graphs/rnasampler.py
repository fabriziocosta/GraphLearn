

from ubergraphlearn import UberSampler
import ubergraphlearn
import networkx as nx
import graphlearn.utils.draw as draw
import random
class RNASampler(UberSampler):

    '''
     the plan is simple, during sampling we work with directed graphs.

     input graphs and cip-graphs graphs are turned into directed graphs before being processed further.
    '''


    '''
    cores and graphs will be turned into diGraphs!
    '''
    def expanded_rna_graph_to_digraph(self,graph):
        '''
        :param graph:  an expanded rna representing graph as produced by eden.
                       properties: backbone edges are replaced by a node labeled '-'.
                       rna reading direction is reflected by ascending node ids in the graph.
        :return: a graph, directed edges along the backbone
        '''
        digraph=nx.DiGraph(graph)
        for n,d in digraph.nodes(data=True):
            if 'edge' in d:
                if d['label']=='-':
                    ns=digraph.neighbors(n)
                    ns.sort()
                    digraph.remove_edge(ns[1],n)
                    digraph.remove_edge(n,ns[0])
        return digraph


    '''
        special stop condition , to make sure that we always work with rna
    '''
    def is_rna (self,graph):
        endcount=0
        for n,d in graph.nodes(data=True):
            if 'node' in d:
                neighbors=graph.neighbors(n)
                backbonecount= len( [ 1 for ne in neighbors if graph.node[ne]['label']=='-' ] )

                if backbonecount == 1:
                    continue
                if backbonecount == 0:
                    endcount+=1
                if backbonecount > 1:
                    print n
                    draw.display(graph,contract=False, size=20,vertex_label='id',vertex_color=None,edge_color='black',node_size=50)
                    raise Exception ('backbone broken')
        return endcount == 1

    def _stop_condition(self, graph):
        self.last_graph=graph.copy()
        if not self.is_rna(graph):
            self._sample_path=[self.last_graph]
            raise Exception('WE CREATED THE ANTI RNA')


    '''
        turning sample starter graph to digraph
    '''
    def _sample_init(self, graph):
        graph = self.vectorizer._edge_to_vertex_transform(graph)
        graph = self.expanded_rna_graph_to_digraph(graph)
        self._score(graph)
        self._sample_notes = ''
        self._sample_path_score_set = set()
        return graph


    '''
        turning cips to diGraph
    '''
    def _select_cips(self, cip):
        for chosen_cip in super(RNASampler,self)._select_cips(cip):
            chosen_cip.graph =  self.expanded_rna_graph_to_digraph(chosen_cip.graph)
            yield chosen_cip

    '''
        this is also used sometimes so we make better sure it doesnt fail
    '''
    def _revert_edge_to_vertex_transform(self,graph):
        # making it to a normal graph before we revert
        graph=nx.Graph(graph)
        return self.vectorizer._revert_edge_to_vertex_transform(graph)



