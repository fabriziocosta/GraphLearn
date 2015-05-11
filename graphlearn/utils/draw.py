import pylab as plt
from eden.util.display import draw_graph

from myeden import *


'''
        
    functions to draw graphs:
        -draw_grammar
            visialize the samplers grammar
        -display(graph)
            use edens drawing thing, but make sure cores and interfaces are marked
        -drawgraphs(list)
            drawing a list of graphs
'''

def graph_clean(graph):

    '''
    in the precess of creating a new graph,
    we marked the nodes that were used as interface and core.
    here we remove the marks.
    :param graph:
    :return:
    '''
    for n, d in graph.nodes(data=True):
        d.pop('root', None)
        d.pop('core', None)
        d.pop('interface', None)

def draw_grammar_stats(grammar):
    c,i,cc,ii=calc_stats_from_grammar(grammar)
    print "how often do we see interfacehashes"
    a=[ (i[k],ii[k]) for k in i.keys()]
    a.sort()
    a0= [e[0] for e in a]
    a1= [e[1] for e in a]
    print 'sum cips: %d' % sum(a0)
    print 'distinct interfaces: %d (seen on x axis)' % len(i)
    print 'y=numberofcores(ihash), y=sumOfCoreCounts(ihash)'
    plt.subplot(1,1,1)
    plt.plot(a0, color='blue', lw=2)
    plt.plot(a1, color='blue', lw=2)
    plt.yscale('log')
    plt.show()

    print 'how often was this corehash seen?'
    a=[ (c[k],cc[k]) for k in c.keys()]
    a.sort()
    a0= [e[0] for e in a]
    a1= [e[1] for e in a]
    print 'sum cips: %d' % sum(a0)
    print 'distinct cores: %d (seen on x axis)' % len(c)
    print 'y = inYinterfaces(chash), y= sumOfCountOverAllInterfaces(chash)'
    plt.subplot(1,1,1)
    plt.plot(a0, color='blue', lw=2)
    plt.plot(a1, color='blue', lw=2)
    plt.yscale('log')
    plt.show()

    print 'histogram'
    #a=[ (c[k],cc[k]) for k in c.keys()]
    a=[ (i[k],ii[k]) for k in i.keys()]
    a.sort()
    a0= [e[0] for e in a]
    d=defaultdict(int)
    for e in a0:
        d[e]+=1
    # [111223] => {1:3 , 2:2  , 3:1}
    datapoints=[]
    for i in range(a0[-1]):
        if i in d:
            datapoints.append(d[i])
        else:
            datapoints.append(0)
    print 'sum cips: %d' % sum(a0)
    print 'distinct cores: %d (seen on x axis)' % len(c)
    print 'interfaces with x many cores were observed y many times. '
    plt.subplot(1,1,1)
    plt.plot(datapoints, color='blue')
    #plt.plot(a1, color='blue', lw=2)
    plt.yscale('log')
    plt.show()



def display(G, size=6, font_size=15, node_size=200, node_border=False, delabeledges=True, contract=False,
            vertex_label='label'):
    if contract:
        G = contract_edges(G)

    G2 = G.copy()


    set_colors(G2)

    if delabeledges:
        for a, b, c in G2.edges_iter(data=True):
            c['label'] = ''

    if vertex_label=='id':
        for n,d in G2.nodes_iter(data=True):
            d['id']=str(n)

    draw_graph(G2, size=size, node_size=node_size, node_border=node_border, font_size=font_size, vertex_color='color',
               vertex_label=vertex_label)



def set_colors(g):
    for n, d in g.nodes(data=True):

        if 'root' in d:
            d['color']='pink'
        elif 'core' in d:
            d['color'] = 'yellow'

        elif 'interface' in d:
            d['color'] = 'green'
        else:
            d['color'] = 'white'

def remove_colors(g):
    for n, d in g.nodes(data=True):
        d['color']='white'

def draw_grammar(grammar, interfacecount):

    # how many rows to draw...
    if len(grammar) < interfacecount:
        interfacecount = len(grammar)

    for i in range(interfacecount):
        interface = grammar.keys()[i]

        core_cid_dict = grammar[interface]

        graphs = [ core_cid_dict[chash].graph for chash in core_cid_dict.keys() ]
        #dists = [core_cid_dict[chash].distance_dict for i, chash in enumerate(core_cid_dict.keys()) if i < 5]

        print 'interface: ' + str(interface)
        draw_many_graphs(graphs)


def cip_to_graph(cips=[],graphs=[]):

    regraphs=[]
    if not graphs:
        for cip in cips:
            graph=cip.graph
            graph.node[cip.distance_dict[0][0]] ['root']=True
            graph.node[cip.distance_dict[0][0]].pop('core')
            regraphs.append(graph)
    else:

        for c,g in zip(cips,graphs):

            remove_colors(g)
            graph_clean(g)
            g2=g.copy()
            d={0:'root'}
            index=1
            for r in range(c.radius-1):
                d[index]='core'
                index+=1
            for t in range(c.thickness):
                d[index]='interface'
                index+=1

            for dist,what in d.items():
                for node_id in c.distance_dict[dist]:
                    g2.node[node_id][what]=True
            regraphs.append(g2)
    return regraphs






def draw_many_graphs(graphs):
    while graphs:
        drawgraphs(graphs[:5])
        graphs=graphs[5:]


def drawgraphs(graphs, contract=True, deleteedges=True, size=4):

    count = len(graphs)
    size_y = size
    size_x = size * 5
    plt.figure(figsize=( size_x, size_y ))
    plt.xlim(xmax=3)


    for x in range(count):
        plt.subplot( 1, 5 , x +1  )
        graphs[x].graph['info']="size:"+str(len(graphs[x]))
        row_drawgraph_wrapper(graphs[x], contract=contract, deleteedges=deleteedges)
    plt.show()





def row_drawgraph_wrapper(G, size=15, font_size=15, node_size=200, node_border=False, contract=True, deleteedges=True):
    if contract:
        G = contract_edges(G)

    if deleteedges:
        for a, b, c in G.edges_iter(data=True):
            c['label'] = ''
    else:
        for a, b, c in G.edges_iter(data=True):
            if 'label' not in c:
                c['label'] = ''

    G2 = G.copy()
    set_colors(G2)
    row_draw_graph(G2, size=size, node_size=node_size, node_border=node_border, font_size=font_size,
                   vertex_color='color')


def row_draw_graph(graph,
                   vertex_label='label',
                   secondary_vertex_label=None,
                   edge_label='label',
                   secondary_edge_label=None,
                   vertex_color='',
                   vertex_alpha=0.6,
                   size=10,
                   size_x_to_y_ratio=1,
                   node_size=600,
                   font_size=9,
                   layout='graphviz',
                   prog='neato',
                   node_border=False,
                   colormap='YlOrRd',
                   invert_colormap=False,
                   verbose=True):
    '''
        thisis basically taken from eden,
        but calling figure() and show() are disables
        so i can draw many graphs in a row
    '''

    plt.grid(False)
    plt.axis('off')

    if secondary_vertex_label:
        vertex_labels = dict(
            [( u, '%s\n%s' % ( d.get(vertex_label, 'N/A'), d.get(secondary_vertex_label, 'N/A')  )  ) for u, d in
             graph.nodes(data=True)])
    else:
        vertex_labels = dict([( u, d.get(vertex_label, 'N/A') ) for u, d in graph.nodes(data=True)])

    edges_normal = [( u, v ) for ( u, v, d ) in graph.edges(data=True) if d.get('nesting', False) == False]
    edges_nesting = [( u, v ) for ( u, v, d ) in graph.edges(data=True) if d.get('nesting', False) == True]

    if secondary_edge_label:
        edge_labels = dict(
            [( ( u, v, ), '%s\n%s' % ( d.get(edge_label, 'N/A'), d.get(secondary_edge_label, 'N/A') )  ) for u, v, d in
             graph.edges(data=True)])
    else:
        edge_labels = dict([( ( u, v, ), d.get(edge_label, 'N/A')  ) for u, v, d in graph.edges(data=True)])

    if vertex_color == '':
        node_color = 'white'
    else:
        if invert_colormap:
            node_color = [- d.get(vertex_color, 0) for u, d in graph.nodes(data=True)]
        else:
            node_color = [d.get(vertex_color, 0) for u, d in graph.nodes(data=True)]

    if layout == 'graphviz':
        pos = nx.graphviz_layout(graph, prog=prog)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'random':
        pos = nx.random_layout(graph)
    elif layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'shell':
        pos = nx.shell_layout(graph)
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph)
    else:
        raise Exception('Unknown layout format: %s' % layout)

    if node_border == False:
        linewidths = 0.001
    else:
        linewidths = 1

    nx.draw_networkx_nodes(graph, pos,
                           node_color=node_color,
                           alpha=vertex_alpha,
                           node_size=node_size,
                           linewidths=linewidths,
                           cmap=plt.get_cmap(colormap)
                           )
    nx.draw_networkx_labels(graph, pos, vertex_labels, font_size=font_size, font_color='black')
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_normal,
                           width=2,
                           edge_color='k',
                           alpha=0.5)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_nesting,
                           width=1,
                           edge_color='k',
                           style='dashed',
                           alpha=0.5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=font_size)
    if verbose:
        title = str(graph.graph.get('id', '')) + "\n" + str(graph.graph.get('info', ''))
        plt.title(title)
        #plt.show()



