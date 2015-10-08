import networkx as nx
from eden.modifier.graph.structure import contraction

from graphlearn.abstract_graphs.rna_graphmanager import get_sequence, getsucc, post
import graphlearn.graph as gt
import graphlearn

'''
direct_abstraction_wrapper
extends the self-build abstractor, but only works on fresh graphs
'''


def direct_abstraction_wrapper(graph,ZZZ):
    """

    :param graph: graph
    :param ZZZ: dummy
    :return: we make the direct abstraction, but merge stuff after..

    1. gimme all the nodes in radius 2  (should be labeled R )

    2. for those nodes check r2 , if they have other friends than us, we can combine them

    """
    abstract_graph= direct_abstractor(graph,None)

    for n,d in abstract_graph.nodes(data=True):
        if d['label']=='B':
            mergelist=[]
            # step 1 get all the "R" arrount us
            neigh=r2_neighbors(abstract_graph,n)
            for ne in neigh:
                if abstract_graph.node[ne]['label']=='R':

            # step 2 check the neighbors of those nodes
                    if len( r2_neighbors(abstract_graph,ne)) > 1:
                        mergelist.append(ne)
            # okok now we need to know who is merging with whoom.
            # good thing, in here we can rely on the node-ids :)
            # remember that edges have higher ids than nucleotides
            for a,b in getpairs(abstract_graph,mergelist,n):
                abstract_graph.node[a]['contracted'].update(abstract_graph.node[b]['contracted'])
                gt.merge(abstract_graph,a,b)
                abstract_graph.node[a]['label']='M'
    return abstract_graph

def getpairs(abstract_graph,mergelist,node):
    low=[]
    high=[]
    """
    the trick here is to get min and max which are from each backbone of the stem.
    """
    stemnodes = list ( abstract_graph.node[node]['contracted'] )
    stemnodes.sort()
    current_min=stemnodes[0]
    for i,n in enumerate( stemnodes):
        if n-i != current_min:
            current_max=n
            break
    for e in mergelist:
        v = min(abstract_graph.node[e]['contracted'])
        if v > current_max or v < current_min:
            high.append(e)
        else:
            low.append(e)

    if len(low) == 2:
        yield low
    if len(high) == 2:
        yield high

def r2_neighbors(graph,n):
    dict=nx.single_source_shortest_path_length(graph, n,2)
    dict=gt.invert_dict(dict)
    if 2 in dict:
        return dict[2]
    else:
        return []








'''
this is the abstractor i built myself.
'''
def direct_abstractor(graph,v):
    '''
    going through the normal channels to make an abstract graph is slow

    here we produce the abstraction directly on the expanded digraph

    :param graph: an expanded digraph.
    :return: contracted graph
    '''

    n,not_used= graphlearn.abstract_graphs.rnaabstract.get_start_and_end_node(graph)
    tasks=[n]
    result = nx.Graph()

    aite={}

    done=set()

    while tasks:
        n=tasks.pop()
        if n in done:
            continue
        stru,out,inc,label = get_substruct(graph,n)
        #print 'recv',stru,out,inc
        if not stru:
            break
        # so the struct gets its own node
        next_node=len(result)
        result.add_node(next_node)
        result.node[next_node]['contracted']=set(stru)
        result.node[next_node]['label']=label
        for o in out:
            # somebody already created the node
            if o in aite:
                out_node = aite[o]
            else:
                out_node=len(result)
                result.add_node(out_node)
                aite[o]=out_node
                result.node[out_node]['contracted']=set([o])
                result.node[out_node]['label']='e'
                result.node[out_node]['edge']=True
            result.add_edge(next_node,out_node)

        for i in inc:
            # somebody already created the node
            if i in aite:
                in_node = aite[i]
            else:
                in_node=len(result)
                result.add_node(in_node)
                aite[i]=in_node
                result.node[in_node]['contracted']=set([i])
                result.node[in_node]['label']='e'
                result.node[in_node]['edge']=True
            result.add_edge(in_node,next_node)

            # dont consider this for calculation. we just calculated this :)
            done.add( graph.neighbors(i)[0])

        tasks+=[graph.neighbors(o)[0] for o in out if graph.neighbors(o)[0] ]

    return result


def predec(graph,root):
    p=graph.predecessors(root)
    for e in p:
        yield e, graph.node[e]


def get_substruct(graph, root):
    '''
    :param graph: the full graph
    :param root: node to start from
    :return: dont know yet(  [ nodes in the structure ],[outgoing],[incoming]   )
    '''
    #print 'entering substr',root
    nei= graph.neighbors(root)
    struct=[root]

    outgoing=[]
    incoming= [n for n,d in predec(graph,root)  if d['label']=='-' ]
    label='R'
    # ok the beginning structure is a stack
    if len(nei)==2:
        label='B'
        # we know there is a bond, we add it.
        backbone,bond = getsucc(graph,root)
        struct+=bond

        # since this is the first one in a stack we have an outgoing edge somewhere, so we can also save that one.
        outgoing+=[n for n,d in post(graph,bond[1]) if d['label']=='-' ]

        while True:
            # now there are 2 possibilities: a stacking or not a stacking
            fail=False
            a = backbone[1]
            abackbone , abond = getsucc(graph,a)
            if not abond:
                fail=True
            else:
                b= abond[1]
                bbackbone , bwhatever = getsucc(graph,b)
                if bbackbone[1]!= struct[-1]:

                    fail=True
            if fail:
                outgoing.append( backbone[0])
                incoming += [n for n,d in predec(graph, bond[1] )  if d['label']=='-' ]
                return struct,outgoing,incoming,label
            else:
                struct+=backbone
                struct.append(bbackbone[0])
                struct+=abond
                backbone=abackbone
                bond = abond
    # the beginning struct is a normal string
    # we start with a nodenode
    elif len(nei)==1:
        while True:
            ne= graph[root].keys()
            if len(ne)==1:
                struct.append(ne[0])
                root = ne[0]
            else:
                return (struct[:-2],[struct[-2]],incoming,label)
    #
    elif len(nei)==0:
        return [[root],[],graph.predecessors(root),label]

    else:
        print 'nei+root',nei,root
        raise Exception('something is wrong with this node')


'''
this is the original abstractor.. it is slow and it makes mistakes.
'''

def annotate(graph):
    '''
    imput: nx.Graph
    ok i got it, its super easy :)
    '''
    attribute='type'
    for n,d in graph.nodes_iter(data=True):
        d[attribute]='R'

    for s,e,d in graph.edges_iter(data=True):
        if d['label'] == '=':
            graph.node[s][attribute]='B'
            graph.node[e][attribute]='B'
    return graph

def contract(graph):
    contracted_graph = contraction(
        [graph], contraction_attribute = 'type', modifiers = [], nesting = False).next()
    return contracted_graph








from eden.converter.rna.rnafold import rnafold_to_eden
class PostProcessor:
    '''
    postprocessotr to refold with rnafold
    '''
    def __init__(self):
        pass

    def fit(self, other):
        print 'OMG i got a vectorizer kthx'
        self.vectorizer=other.vectorizer

    def postprocess(self, graph):
        return self.rna_refold( graph )

    def rna_refold(self, digraph=None, seq=None,vectorizer=None):
        """
        :param digraph:
        :param seq:
        :return: will extract a sequence, RNAfold it and create a abstract graph
        """
        # get a sequence no matter what :)
        if not seq:
            seq= get_sequence(digraph)
        #print 'seq:',seq
        graph = rnafold_to_eden([('emptyheader',seq)], shape_type=5, energy_range=30, max_num=3).next()
        expanded_graph = self.vectorizer._edge_to_vertex_transform(graph)
        ex_di_graph = graphlearn.abstract_graphs.rnaabstract.expanded_rna_graph_to_digraph(expanded_graph)
        ex_di_graph.graph['sequence']= seq
        #abstract_graph = directedgraphtools.direct_abstraction_wrapper(graph,0)
        return ex_di_graph



