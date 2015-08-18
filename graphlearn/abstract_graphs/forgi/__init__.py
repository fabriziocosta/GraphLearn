#forgi is here: https://github.com/pkerpedjiev/forgi
#i took what i needed and removed unnecessary dependencies
# in this file i try to make a wrapper 


'''
TODO:
-create graph, has abs graph as attribute.
-merge function
-ubersampler muss verstehen dass er die abstrakte form lernen muss
'''

import networkx as nx
import graphlearn.abstract_graphs.forgi.bulge_graph as lol


def get_abstr_graph(struct):
    # get forgi string
    bg = lol.BulgeGraph()
    bg.from_dotbracket(struct, None)
    forgi = bg.to_bg_string()

    g=make_abstract_graph(forgi)
    return g


def make_abstract_graph(forgi):
    g=forgi_to_graph(forgi)
    connect_multiloop(g)
    return g


def forgi_to_graph(forgi):
    def make_node_set(numbers):
        '''
        forgi gives me stuff like define STEM START,END,START,END .. we take indices and output a list
        '''
        numbers=map(int,numbers)
        ans=set()
        while len(numbers)>1:
            a,b = numbers[:2]
            numbers=numbers[2:]
            for n in range(a-1,b): ans.add(n) # should be range a,b+1 but the biologists are weired
        return ans


    def get_pairs(things):
        '''
        '''
        current=[]
        for thing in things:
            if thing[0]=='m':
                current.append(thing)
            if len(current)==2:
                yield current
                current = []


    g=nx.Graph()
    fni={} # forgi name to networkx node id

    for l in forgi.split('\n')[:-1]:
        line= l.split()
        if line[0] not in ['define','connect']:
            continue

        # parse stuff like: define s0 1 7 65 71
        if line[0]=='define':

            # get necessary attributes for a node
            label=line[1][0]
            id=line[1]
            myset=make_node_set(line[2:])
            node_id=len(g)

            # build a node and remember its id
            g.add_node(node_id)
            fni[id]=node_id
            g.node[node_id].update( {'label':label, 'contracted':myset}  )

        # parse stuff like this: connect s3 h2 m1 m3
        if line[0]=='connect':
            # get nx name of the first element.
            hero= fni[ line[1] ]
            # connect that hero to the following elements
            for fn in line[2:]:
                g.add_edge(hero, fni[fn])

            # remember what pairs multiloop pieces we are part of
            # i assume that if a stack is part of 2 multiloops they appear in order ..
            # this assumption may be wrong so be careful
            g.node[fni[line[1]]]['multipairs']=[]
            for a,b in get_pairs(line[2:]):
                g.node[fni[line[1]]]['multipairs'].append( (fni[a],fni[b]) )
    return g


def connect_multiloop(g):

    def merge(graph, node, node2):
        '''
        merge node2 into the node.
        input nodes are strings,
        node is the king
        '''
        for n in graph.neighbors(node2):
            graph.add_edge(node, n)
        graph.node[node]['contracted'].update(graph.node[node2]['contracted'])
        graph.remove_node(node2)


    merge_dict={}
    for node,d in g.nodes(data=True):
        if d['label'] == 's':
            for a,b in g.node[node]['multipairs']:
                # finding real names... this works by walking up the
                #ladder merge history until the root is found :)
                while a not in g:
                    a=merge_dict[a]
                while b not in g:
                    b=merge_dict[b]
                if a==b:
                    continue
                merge_dict[b]=a
                merge(g,a,b)

