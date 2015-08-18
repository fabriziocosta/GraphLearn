

'''
graphs can have so many states.
they can be expanded, refolded, directed

i need some order in this.
this is why i created the graph manager.
it just manages a single graph.
'''
import subprocess  as sp
import eden.converter.rna as conv
import forgi
import networkx as nx
from eden.graph import Vectorizer

import graphlearn.abstract_graphs.rnasampler as rna

def fromfasta(file='RF00005.fa'):
    s=[]
    with open('RF00005.fa') as f:
        for line in f:
            s.append(line)

    while s:
        seqname=s[0]
        seq=''
        for i,l in enumerate(s[1:]):
            if l[0] != '>':
                seq+=l
            else:
                break
        yield GraphManager(seqname,seq)
        s=s[i+1:]




class GraphManager(object):

    def __init__(self,sequence_name,sequence,vectorizer):
        self.sequence_name=sequence_name
        self.sequence=sequence
        self.vectorizer=vectorizer

        shape=self.callRNAshapes(sequence)

        if shape is None:
            return
        self.structure=shape


        # create a base_graph , expand
        base_graph=conv.sequence_dotbracket_to_graph(seq_info=sequence_name, seq_struct=shape)
        self.base_graph=vectorizer._edge_to_vertex_transform(base_graph)

        # get an expanded abstract graph
        abstract_graph=forgi.get_abstr_graph(shape)
        abstract_graph=vectorizer._edge_to_vertex_transform(abstract_graph)

        #connect edges to nodes in the abstract graph
        self.abstract_graph=edge_parent_finder(abstract_graph,base_graph)



    def callRNAshapes(self,sequence):

        cmd='RNAshapes %s' % sequence
        out = sp.check_output(cmd, shell=True)
        s = out.strip().split('\n')

        for li in s[2:]:
            #print li.split()
            energy,shape,abstr= li.split()
            if abstr=='[[][][]]':
                return shape


    def get_estimateable(self):

        # returns an expanded, undirected graph
        # that the eden machine learning can compute
        return nx.disjoint_union(self.base_graph,self.abstract_graph)

    def get_grammar_input(self):
        # return something the can be used for the grammar.
        graph=rna.expanded_rna_graph_to_digraph(self.base_graph)

        return graph









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
