''' utility functions to deal with rna '''
import networkx as nx



def textwrap(seq, width=60):
    '''
    "asdasdasd"-> "asd" "asd" "asd"
    '''
    res=[]
    while len(seq) > width:
        #print seq
        res.append(seq[:width])
        seq=seq[width:]
    res.append(seq)
    return res

def write_fasta(sequences, filename='asdasd'):
    '''
    write fastasequences to file
    '''

    fasta = ''
    for i, s in enumerate(sequences):
        if len(s) > 5:
            seq = s.replace("F", "")
            if not is_sequence(seq):
                continue
            seq = '\n'.join(textwrap(seq, width=60))
            fasta += '>HACK%d\n%s\n\n' % (i, seq)
    with open(filename, 'w') as f:
        f.write(fasta)


def _pairs(dotbracket):
    "dotbracket string to bond dictionary"
    unpaired = []
    pairs = {}
    for i, c in enumerate(dotbracket):
        if c == '(':
            unpaired.append(i)
        if c == ')':
            partner = unpaired.pop()
            pairs[i] = partner
            pairs[partner] = i
    return pairs




def expanded_rna_graph_to_digraph(graph):
    '''
    :param graph:  an expanded rna representing graph as produced by eden.
                   properties: backbone edges are replaced by a node labeled '-'.
                   rna reading direction is reflected by ascending node ids in the graph.
    :return: a graph, directed edges along the backbone
    '''
    digraph = nx.DiGraph(graph)
    for n, d in digraph.nodes(data=True):
        if 'edge' in d:
            if d['label'] == '-':
                ns = digraph.neighbors(n)
                ns.sort()
                digraph.remove_edge(ns[1], n)
                digraph.remove_edge(n, ns[0])
    return digraph



def get_sequence(digraph):
    '''
    graph: nx.digraph
        the digraph represents an rna string, as such there is a sequence that we can read
        along the backbone
    returns: (id_of_start_node,id_of_endnode)
    '''
    if type(digraph) == str:
        return digraph
    #from graphlearn.utils import draw
    #draw.debug(digraph)
    current, end = get_start_and_end_node(digraph)
    seq = digraph.node[current]['label']
    while current != end:
        current = _getsucc(digraph, current)[0][1]
        seq += digraph.node[current]['label']
    return seq


def _getsucc(graph, root):
    '''
    :param graph:
    :param root:
    :return: [ edge node , nodenode ] along the 'right' path   [edge node, nodenode  ] along the wroong path
    '''

    def post(graph, root):
        p = graph.neighbors(root)
        for e in p:
            yield e, graph.node[e]

    neighbors = post(graph, root)
    retb = []
    reta = []

    for node, dict in neighbors:
        if dict['label'] == '-':
            reta.append(node)
            reta += graph[node].keys()

        if dict['label'] == '=':
            retb.append(node)
            retb += graph[node].keys()
            retb.remove(root)

    # print 'getsuc',reta, retb,root
    return reta, retb

def get_start_and_end_node(graph):
    '''
    graph: nx.digraph
        the digraph represents an rna string, as such has a start and an end node.
    returns: (id_of_start_node,id_of_endnode)
    '''

    start = -1
    end = -1
    for n, d in graph.nodes_iter(data=True):

        # edge nodes cant be start or end
        if 'edge' in d:
            continue

        # check for start
        if start == -1:
            l = graph.predecessors(n)
            if len(l) == 0:
                start = n
            if len(l) == 1:
                if graph.node[l[0]]['label'] == '=':
                    start = n

        # check for end:
        if end == -1:
            l = graph.neighbors(n)
            if len(l) == 0:
                end = n
            if len(l) == 1:
                if graph.node[l[0]]['label'] == '=':
                    end = n

    # check and return
    if start == -1 or end == -1:
        #import graphlearn.utils.draw as draw
        #draw.graphlearn(graph)
        raise Exception('your beautiful "rna" has no clear start or end')

    return start, end


def is_sequence(seq):
    nuc = ["A", "U", "C", "G", "N"]  # :)  some are N in the rfam fasta file.
    for e in seq:
        if e not in nuc:
            return False
    return len(seq) > 5
