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


def display(G, size=15, font_size=15, node_size=200, node_border=False, delabeledges=True, contract=False,
            vertex_label='label'):
    if contract:
        G = contract_edges(G)

    G2 = G.copy()
    for n, d in G2.nodes(data=True):
        if 'core' in d:
            d['color'] = 'red'
        elif 'interface' in d:
            d['color'] = 'yellow'
        else:
            d['color'] = 'green'
    if delabeledges:
        for a, b, c in G2.edges_iter(data=True):
            c['label'] = ''

    draw_graph(G2, size=size, node_size=node_size, node_border=node_border, font_size=font_size, vertex_color='color',
               vertex_label=vertex_label)


def draw_grammar(sampler, interfacecount):
    grammar = sampler.substitute_grammar
    # how many rows to draw...
    if len(grammar) < interfacecount:
        interfacecount = len(grammar)

    for i in range(interfacecount):
        interface = grammar.keys()[i]

        core_cid_dict = grammar[interface]

        graphs = [core_cid_dict[chash].graph for i, chash in enumerate(core_cid_dict.keys()) if i < 5]

        print 'interface: ' + str(interface)
        drawgraphs(graphs, len(core_cid_dict))


def drawgraphs(graphs, contract=True, deleteedges=True, size=4):
    count = len(graphs)
    size_y = size
    size_x = size_y * count
    plt.figure(figsize=( size_x, size_y ))
    plt.xlim(xmax=3)

    for x in range(count):
        plt.subplot(1, count, x + 1)
        row_drawgraph_wrapper(graphs[x], contract=contract, deleteedges=deleteedges)

    plt.show()


def row_drawgraph_wrapper(G, size=15, font_size=15, node_size=200, node_border=False, contract=True, deleteedges=True):
    if contract:
        G = contract_edges(G)

    if deleteedges:
        for a, b, c in G.edges_iter(data=True):
            c['label'] = ''

    for a, b, c in G.edges_iter(data=True):
        if 'label' not in c:
            c['label'] = ''

    G2 = G.copy()
    for n, d in G2.nodes(data=True):
        if 'core' in d:
            d['color'] = 'blue'
        elif 'interface' in d:
            d['color'] = 'pink'
        else:
            d['color'] = 'yellow'

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

    size_x = size
    size_y = int(float(size) / size_x_to_y_ratio)

    # plt.figure( figsize = ( size_x,size_y ) )
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
                           cmap=plt.get_cmap(colormap))
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



