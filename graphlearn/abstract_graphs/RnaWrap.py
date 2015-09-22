from ubergraphlearn import UberGraphWrapper
import eden
import networkx as nx
import subprocess as sp
import forgi
import eden.converter.rna as converter

def GraphWrapper(base_thickness_list=[2]):
    return lambda x,y:RnaGraphWrapper(x,y,base_thickness_list=base_thickness_list)


class RnaGraphWrapper(UberGraphWrapper):

    def abstract_graph(self):
        '''
        we need to make an abstraction Ooo
        '''
        if self._abstract_graph == None:

            # create the abstract graph and populate the contracted set
            abstract_graph = forgi.get_abstr_graph(self.structure)
            abstract_graph = self.vectorizer._edge_to_vertex_transform(abstract_graph)
            self._abstract_graph = edge_parent_finder(abstract_graph, self._base_graph)


            #eden is forcing us to set a label and a contracted attribute.. lets do this
            for n, d in self._abstract_graph.nodes(data=True):
                if 'edge' in d:
                    d['label'] = 'e'
            # in the abstract graph , all the edge nodes need to have a contracted attribute.
            # originaly this happens naturally but since we make multiloops into one loop there are some left out
            for n, d in self._abstract_graph.nodes(data=True):
                if 'contracted' not in d:
                    d['contracted'] = set()


        return self._abstract_graph



    def __init__(self,graph,vectorizer=eden.graph.Vectorizer(), base_thickness_list=None):
        '''
        we need to do some folding here
        '''



        self.some_thickness_list=base_thickness_list
        self.vectorizer=vectorizer
        self._abstract_graph= None
        self._base_graph=graph


        if len(graph) > 0:
            try:
                self.sequence = get_sequence(graph)
            except:
                from graphlearn.utils import draw
                print 'sequenceproblem:'
                draw.graphlearn(graph, size=20)
            self.structure = callRNAshapes(self.sequence)
            self._base_graph = converter.sequence_dotbracket_to_graph(seq_info=self.sequence, seq_struct=self.structure)
            self._base_graph = vectorizer._edge_to_vertex_transform(self._base_graph)
            self._base_graph = expanded_rna_graph_to_digraph(self._base_graph)

            # normaly anything in the core can be replaced,
            # the mod dict is a way arrounf that rule.. it allows to mark special nodes that can only
            # be replaced by something having the same marker.
            # we dont want start and end nodes to disappear, so we mark them :)
            s,e= get_start_and_end_node(self.base_graph())
            self._mod_dict= {s:696969 , e:123123123}



def edge_parent_finder(abstract, graph):
    # find out to which abstract node the edges belong
    # finding out where the edge-nodes belong, because the contractor cant possibly do this
    #draw.graphlearn_draw([abstract,graph],size=10, contract=False,vertex_label='id')

    getabstr = {contra: node for node, d in abstract.nodes(data=True) for contra in d.get('contracted', [])}
    # print getabstr
    for n, d in graph.nodes(data=True):
        if 'edge' in d:
            # if we have found an edge node...

            # lets see whos left and right of it:
            # if len is 2 then we hit a basepair, in that case we already have both neighbors
            zomg = graph.neighbors(n)
            if len(zomg)==1:
                zomg+=graph.predecessors(n)

            n1, n2 = zomg


            # case1: ok those belong to the same gang so we most likely also belong there.
            if getabstr[n1] == getabstr[n2]:
                abstract.node[getabstr[n1]]['contracted'].add(n)

            # case2: neighbors belong to different gangs...
            else:
                abstract_intersect = set(abstract.neighbors(getabstr[n1])) & set(abstract.neighbors(getabstr[n2]))

                # case 3: abstract intersect in radius 1 failed, so lets try radius 2
                if not abstract_intersect:
                    abstract_intersect = set(nx.single_source_shortest_path(abstract, getabstr[n1], 2)) & set(
                        nx.single_source_shortest_path(abstract, getabstr[n2], 2))
                    if len(abstract_intersect) > 1:
                        print "weired abs intersect..."

                for ai_node in abstract_intersect:
                    if 'contracted' in abstract.node[ai_node]:
                        abstract.node[ai_node]['contracted'].add(n)
                    else:
                        abstract.node[ai_node]['contracted'] = set([n])

    return abstract



def get_sequence(digraph):
    if type(digraph)==str:
        return digraph
    current,end= get_start_and_end_node(digraph)
    seq=digraph.node[current]['label']
    while current != end:
        current = _getsucc(digraph,current)[0][1]

        seq+=digraph.node[current]['label']
    return seq

def _getsucc(graph,root):
    '''
    :param graph:
    :param root:
    :return: [ edge node , nodenode ] along the 'right' path   [edge node, nodenode  ] along the wroong path
    '''
    def post(graph,root):
        p=graph.neighbors(root)
        for e in p:
            yield e, graph.node[e]

    neighbors=post(graph,root)
    retb=[]
    reta=[]

    for node,dict in neighbors:
        if dict['label'] == '-':
            reta.append(node)
            reta+=graph[node].keys()

        if dict['label'] == '=':
            retb.append(node)
            retb+=graph[node].keys()
            retb.remove(root)

    #print 'getsuc',reta, retb,root
    return reta, retb




def get_start_and_end_node(graph):
    # make n the first node of the sequence
    start=-1
    end=-1
    for n,d in graph.nodes_iter(data=True):

        # edge nodes cant be start or end
        if 'edge' in d:
            continue

        # check for start
        if start == -1:
            l= graph.predecessors(n)
            if len(l)==0:
                start = n
            if len(l)==1:
                if graph.node[ l[0] ]['label']=='=':
                    start = n

        # check for end:
        if end == -1:
            l= graph.neighbors(n)
            if len(l)==0:
                end = n
            if len(l)==1:
                if graph.node[ l[0] ]['label']=='=':
                    end = n

    # check and return
    if start==-1 or end==-1:
        raise Exception ('your beautiful "rna" has no clear start or end')
    return start,end


def expanded_rna_graph_to_digraph(graph):
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



def is_rna (graph):
    graph=graph.copy()
    # remove structure
    bonds= [ n for n,d in graph.nodes(data=True) if d['label']=='=' ]
    graph.remove_nodes_from(bonds)
    # see if we are cyclic
    for node,degree in graph.in_degree_iter( graph.nodes() ):
        if degree == 0:
            break
    else:
        return False
    # check if we are connected.
    graph=nx.Graph(graph)
    return nx.is_connected(graph)



def callRNAshapes(sequence):

    cmd = 'RNAshapes %s' % sequence
    out = sp.check_output(cmd, shell=True)
    s = out.strip().split('\n')

    for li in s[2:]:
        # print li.split()
        energy, shape, abstr = li.split()
        #if abstr == '[[][][]]':
        return shape



