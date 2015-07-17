
from ubergraphlearn import UberSampler,UberGrammar
import ubergraphlearn
import networkx as nx
import graphlearn.utils.draw as draw
import random
from eden.modifier.graph.structure import contraction


class RNASampler(UberSampler):

    def _propose(self, graph):
        '''
         we wrap the propose single cip, so it may be overwritten some day
        '''


        graph2 = None
        while graph2 == None:
            graph2 = self._propose_graph(graph)

        return graph2

    '''
    def _stop_condition(self, graph):
        self.last_graph=graph.copy()
        if not is_rna(graph):
            self._sample_path=[self.last_graph]
            raise Exception('WE CREATED THE ANTI RNA')
    '''

    '''
        turning sample starter graph to digraph
    '''
    def _sample_init(self, graph):
        graph = self.vectorizer._edge_to_vertex_transform(graph)
        graph = expanded_rna_graph_to_digraph(graph)
        self._score(graph)
        self._sample_notes = ''
        self._sample_path_score_set = set()
        return graph


    '''
        this is also used sometimes so we make better sure it doesnt fail
    '''
    def _revert_edge_to_vertex_transform(self,graph):
        # making it to a normal graph before we revert
        graph=nx.Graph(graph)
        try:
            graph=self.vectorizer._revert_edge_to_vertex_transform(graph)
            return graph
        except:
            print 'revert broke'
            draw.display(graph,contract=False)



    def __init__(self,**kwargs):
        super(RNASampler, self).__init__(**kwargs)
        self.feasibility_checker.checklist.append(is_rna)


'''
    rna checker
'''
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



#ubergraphlearn.arbitrary_graph_abstraction_function = lambda x: contract(annotate(x))


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




def direct_abstractor(graph,v):
    '''
    going through the normal channels to make an abstract graph is slow

    here we produce the abstraction directly on the expanded digraph

    :param graph: an expanded digraph.
    :return: contracted graph
    '''
    # make n the first node of the sequence
    for n,d in graph.nodes_iter(data=True):
        if 'edge' not in d:
            l= graph.predecessors(n)
            if len(l)==0:
                break
            if len(l)==1:
                if graph.node[ l[0] ]['label']=='=':
                    break
    if n==len(graph):
        raise Exception ('cant make this abstract, its not an rna, go away!')

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



ubergraphlearn.make_abstract = direct_abstractor


def predec(graph,root):
    p=graph.predecessors(root)
    for e in p:
        yield e, graph.node[e]

def post(graph,root):
    p=graph.neighbors(root)
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



def getsucc(graph,root):
    '''
    :param graph:
    :param root:
    :return: [ edge node , nodenode ] along the 'right' path   [edge node, nodenode  ] along the wroong path
    '''
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


