'''
ubersamplers fit will take a list of graphmanager as input.
'''

import subprocess  as sp
import eden.converter.rna as conv
import forgi
import networkx as nx
import graphlearn.abstract_graphs.rnaabstract
from graphlearn.utils import draw
from eden.graph import Vectorizer


import rnasampler as rna
import rnaabstract as rnaa

def fromfasta(file='RF00005.fa',vectorizer=None):
    s=[]
    with open('RF00005.fa') as f:
        for line in f:
            s.append(line)

    while s:
        seqname=s[0]
        seq=''
        for i,l in enumerate(s[1:]):
            if l[0] != '>':
                seq+=l.strip()
            else:
                break

        shape=callRNAshapes(seq)
        if shape:
            yield GraphManager(seqname,seq,vectorizer,shape)
        s=s[i+1:]


class GraphManager(object):
    '''
    these are the basis for creating a fitting an ubersampler
    def get_estimateable(self):
    def get_base_graph(self):
    def get_abstract_graph(self):

    '''
    def __init__(self,sequence_name,sequence,vectorizer,structure):
        self.sequence_name=sequence_name
        self.sequence=sequence
        self.vectorizer=vectorizer
        self.structure=structure

        # create a base_graph , expand
        base_graph=conv.sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=structure)
        self.base_graph=vectorizer._edge_to_vertex_transform(base_graph)

        # get an expanded abstract graph
        abstract_graph=forgi.get_abstr_graph(structure)
        abstract_graph=vectorizer._edge_to_vertex_transform(abstract_graph)

        #connect edges to nodes in the abstract graph
        self.abstract_graph=edge_parent_finder(abstract_graph,self.base_graph)


        # we are forced to set a label .. for eden reasons
        def name_edges(graph,what=''):
            for n,d in graph.nodes(data=True):
                if 'edge' in d:
                    d['label']=what

        # in the abstract graph , all the edge nodes need to have a contracted attribute.
        # originaly this happens naturally but since we make multiloops into one loop there are some left out
        def setset(graph):
            for n,d in graph.nodes(data=True):
                if 'contracted' not in d:
                    d['contracted']=set()

        name_edges(self.abstract_graph)
        setset(self.abstract_graph)


    def get_estimateable(self):

        # returns an expanded, undirected graph
        # that the eden machine learning can compute
        return nx.disjoint_union(self.base_graph,self.abstract_graph)

    def get_base_graph(self):
        if 'directed_base_graph' not in self.__dict__:
            self.directed_base_graph= graphlearn.abstract_graphs.rnaabstract.expanded_rna_graph_to_digraph(self.base_graph)

        return self.directed_base_graph

    def get_abstract_graph(self):
        return self.abstract_graph






def callRNAshapes(sequence):

    cmd='RNAshapes %s' % sequence
    out = sp.check_output(cmd, shell=True)
    s = out.strip().split('\n')

    for li in s[2:]:
        #print li.split()
        energy,shape,abstr= li.split()
        if abstr=='[[][][]]':
            return shape


def edge_parent_finder(abstract,graph):
    # find out to which abstract node the edges belong
    # finding out where the edge-nodes belong, because the contractor cant possibly do this
    #draw.graphlearn_draw([abstract,graph],size=10, contract=False,vertex_label='id')

    getabstr = {contra: node for node, d in abstract.nodes(data=True) for contra in d.get('contracted', [])}
    #print getabstr
    for n, d in graph.nodes(data=True):
        if 'edge' in d:
            # if we have found an edge node...
            # lets see whos left and right of it:
            zomg= graph.neighbors(n)
            #print zomg
            #draw.draw_center(graph,1,20,contract=False,size=20)
            n1, n2 = zomg

            # case1: ok those belong to the same gang so we most likely also belong there.
            if getabstr[n1] == getabstr[n2]:
                abstract.node[getabstr[n1]]['contracted'].add(n)

            # case2: neighbors belong to different gangs...
            else:
                abstract_intersect = set(abstract.neighbors(getabstr[n1])) & set(abstract.neighbors(getabstr[n2]))

                # case 3: abstract intersect in radius 1 failed, so lets try radius 2
                if not abstract_intersect:
                    abstract_intersect = set(nx.single_source_shortest_path(abstract,getabstr[n1],2)) & set(nx.single_source_shortest_path(abstract,getabstr[n2],2))
                    if len(abstract_intersect) > 1:
                        print "weired abs intersect..."

                for ai_node in abstract_intersect:
                    if 'contracted' in abstract.node[ai_node]:
                        abstract.node[ai_node]['contracted'].add(n)
                    else:
                        abstract.node[ai_node]['contracted'] = set([n])


    return abstract
