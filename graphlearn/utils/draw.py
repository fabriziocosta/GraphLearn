import pylab as plt
from eden.util.display import draw_graph
import networkx as nx
import graphlearn.graphtools as graphtools
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


'''
        
    functions to draw graphs:
        -draw_grammar
            visialize the samplers grammar
        -display(graph)
            use edens drawing thing, but make sure cores and interfaces are marked
        -draw_graphs(list)
            drawing a list of graphs
'''



def plot_charts(data1, data2=None, xlabel=None, ylabel=None, size=(10,4), log_scale=True):
    plt.figure(figsize=size)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data1, color='blue', lw=2)
    plt.plot(data1, linestyle='None', markerfacecolor='white', markeredgecolor='blue', marker='o', markeredgewidth=2, markersize=8)
    if data2 is not None:
        plt.plot(data2, color='red', lw=2)
        plt.plot(data2, linestyle='None', markerfacecolor='white', markeredgecolor='red', marker='o', markeredgewidth=2, markersize=8)
    if log_scale:
        plt.yscale('log')
    plt.xlim(-0.2, len(data1) + 0.2)
    plt.ylim(0.8)
    plt.show()


def draw_grammar_stats(grammar, size=(10,4)):
    c, i, cc, ii = calc_stats_from_grammar(grammar)
    print "how often do we see interfacehashes"
    a = [(i[k], ii[k]) for k in i.keys()]
    a.sort()
    a0 = [e[0] for e in a]
    a1 = [e[1] for e in a]
    print '# productions: %d' % sum(a0)

    print 'x = # interfaces (total: %d)' % len(i)
    print 'y=numberofcores(ihash), y=sumOfCoreCounts(ihash)'
    plot_charts(a0, a1, xlabel='# interfaces', ylabel='counts', size=size)

    print 'how often was this corehash seen?'
    a = [(c[k], cc[k]) for k in c.keys()]
    a.sort()
    a0 = [e[0] for e in a]
    a1 = [e[1] for e in a]
    print 'x = # cores (total: %d)' % len(c)
    print 'y = inYinterfaces(chash), y= sumOfCountOverAllInterfaces(chash)'
    plot_charts(a0, a1, xlabel='# cores', ylabel='counts', size=size)

    print 'histogram'
    #a=[ (c[k],cc[k]) for k in c.keys()]
    a = [(i[k], ii[k]) for k in i.keys()]

    a0 = [e[0] for e in a]
    a0.sort()

    d = defaultdict(int)
    for e in a0:
        d[e] += 1
    # [111223] => {1:3 , 2:2  , 3:1}
    datapoints = []
    for i in range(a0[-1]):
        if i in d:
            datapoints.append(d[i])
        else:
            datapoints.append(0)
    # datapoints.sort()
    print '# productions: %d' % sum(a0)
    print 'distinct cores: %d (seen on x axis)' % len(c)
    print 'interfaces with x many cores were observed y many times. '
    plot_charts(datapoints, size=size)

    print 'other histogram'
    print 'how many cores exist with x many interfaces'
    nc = [ v for v in c.values()  ]
    nc.sort()
    d=defaultdict(int)
    for e in nc:
        d[e]+=1
    dp=[]
    for i in range(  max(nc)  ):
        if i in d:
            dp.append(d[i])
        else:
            dp.append(d[i])

    plot_charts(dp,size=size)




def display(G, size=6, font_size=15, node_size=200, node_border=False, delabeledges=True, contract=False, vertex_label='label'):
    if contract:
        G = contract_edges(G)
    G2 = G.copy()
    set_colors(G2)
    if delabeledges:
        for a, b, c in G2.edges_iter(data=True):
            c['label'] = ''
    if vertex_label == 'id':
        for n, d in G2.nodes_iter(data=True):
            d['id'] = str(n)
    draw_graph(G2, size=size, node_size=node_size, node_border=node_border, font_size=font_size, vertex_color='color', vertex_label=vertex_label)


def cip_to_graph(cips=[], graphs=[]):
    regraphs = []
    if not graphs:
        for cip in cips:
            graph = cip.graph
            graph.node[cip.distance_dict[0][0]]['root'] = True
            graph.node[cip.distance_dict[0][0]].pop('core')
            regraphs.append(graph)
    else:
        for c, g in zip(cips, graphs):
            remove_colors(g)
            graphtools.graph_clean(g)
            g2 = g.copy()
            d = {0: 'root'}
            index = 1
            for r in range(c.radius - 1):
                d[index] = 'core'
                index += 1
            for t in range(c.thickness):
                d[index] = 'interface'
                index += 1

            for dist, what in d.items():
                for node_id in c.distance_dict[dist]:
                    g2.node[node_id][what] = True
            regraphs.append(g2)
    return regraphs


def set_colors(g, key='col'):
    for n, d in g.nodes(data=True):
        if 'root' in d:
            d[key] = 1
        elif 'core' in d:
            d[key] = 0.65
        elif 'interface' in d:
            d[key] = 0.45
        else:
            d[key] = 0


def remove_colors(g, key='col'):
    for n, d in g.nodes(data=True):
        d[key] = 'white'


def draw_grammar(grammar, n_productions=None, n_graphs_per_line=5, size=4, **args):
    if n_productions is None:
        n_productions = len(grammar)

    if len(grammar) < n_productions:
        n_productions = len(grammar)

    for i in range(n_productions):
        interface = grammar.keys()[i]
        core_cid_dict = grammar[interface]
        graphs = [core_cid_dict[chash].graph for chash in core_cid_dict.keys()]
        #dists = [core_cid_dict[chash].distance_dict for i, chash in enumerate(core_cid_dict.keys()) if i < 5]
        print 'interface: ' + str(interface)
        draw_graph_set(graphs, n_graphs_per_line=n_graphs_per_line, size=size, **args)


def draw_graph_set(graphs, n_graphs_per_line=5, size=4, **args):
    graphs=list(graphs)
    while graphs:
        draw_graphs(graphs[:n_graphs_per_line], n_graphs_per_line=n_graphs_per_line, size=size, **args)
        graphs = graphs[n_graphs_per_line:]


def draw_graphs(graphs, contract=True, delete_edges=True, n_graphs_per_line=5, size=4, vertex_color=None, **args):
    count = len(graphs)
    size_y = size
    size_x = size * n_graphs_per_line
    plt.figure(figsize=(size_x, size_y))
    plt.xlim(xmax=3)

    for i in range(count):
        plt.subplot(1, n_graphs_per_line, i + 1)
        graphs[i].graph['info'] = "size:" + str(len(graphs[i]))
        draw_graph_row_wrapper(graphs[i], n_graphs_per_line=n_graphs_per_line, contract=contract, delete_edges=delete_edges, vertex_color=vertex_color, **args)
    plt.show()


def draw_graph_row_wrapper(G, size=15, font_size=15, node_size=200, node_border=False, contract=True, delete_edges=True, vertex_color=None, **args):
    if contract:
        G = contract_edges(G)
    if delete_edges:
        for a, b, c in G.edges_iter(data=True):
            c['label'] = ''
    else:
        for a, b, c in G.edges_iter(data=True):
            if 'label' not in c:
                c['label'] = ''
    g = G.copy()
    if vertex_color is None:
        set_colors(g)
        vertex_color='col'
    draw_graph_row(g, size=size, node_size=node_size, node_border=node_border, font_size=font_size, vertex_color=vertex_color, **args)


def draw_graph_row(graph,
                   vertex_label='label',
                   secondary_vertex_label=None,
                   edge_label='label',
                   secondary_edge_label=None,
                   vertex_color='',
                   vertex_alpha=0.6,
                   edge_alpha=0.5,
                   node_size=600,
                   font_size=9,
                   layout='graphviz',
                   prog='neato',
                   node_border=False,
                   colormap='YlOrRd',
                   invert_colormap=False,
                   verbose=True,
                   **args):
    '''
        thisis basically taken from eden,
        but calling figure() and show() are disables
        so i can draw many graphs in a row
    '''

    plt.grid(False)
    plt.axis('off')

    if secondary_vertex_label:
        vertex_labels = dict(
            [(u, '%s\n%s' % (d.get(vertex_label, 'N/A'), d.get(secondary_vertex_label, 'N/A'))) for u, d in
             graph.nodes(data=True)])
    else:
        vertex_labels = dict([(u, d.get(vertex_label, 'N/A')) for u, d in graph.nodes(data=True)])

    edges_normal = [(u, v) for (u, v, d) in graph.edges(data=True) if d.get('nesting', False) == False]
    edges_nesting = [(u, v) for (u, v, d) in graph.edges(data=True) if d.get('nesting', False) == True]

    if secondary_edge_label:
        edge_labels = dict(
            [((u, v, ), '%s\n%s' % (d.get(edge_label, 'N/A'), d.get(secondary_edge_label, 'N/A'))) for u, v, d in
             graph.edges(data=True)])
    else:
        edge_labels = dict([((u, v, ), d.get(edge_label, 'N/A')) for u, v, d in graph.edges(data=True)])

    if vertex_color == '':
        node_color = 'white'
    elif vertex_color == '_labels_':
        node_color = [hash(d.get('label', '.')) & 15 for u, d in graph.nodes(data=True)]
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
                           alpha=edge_alpha)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_nesting,
                           width=1,
                           edge_color='k',
                           style='dashed',
                           alpha=edge_alpha)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=font_size)
    if verbose:
        title = str(graph.graph.get('id', '')) + "\n" + str(graph.graph.get('info', ''))
        plt.title(title)
        # plt.show()


def calc_stats_from_grammar(grammar):
    count_corehashes = defaultdict(int)
    count_interfacehashes = defaultdict(int)
    corecounter = defaultdict(int)
    intercounter = defaultdict(int)
    for ih in grammar.keys():
        for ch in grammar[ih].keys():
            # go over all the combos
            count_corehashes[ch]+=1
            count_interfacehashes[ih]+=1
            count= grammar[ih][ch].count
            corecounter[ch]+=count
            intercounter[ih]+=count
    return count_corehashes,count_interfacehashes,corecounter,intercounter

def contract_edges(original_graph):
    """
        stealing from eden...
        because i draw cores and interfaces there may be edge-nodes
        that have no partner, eden gives error in this case.
        i still want to see them :)
    """
    # start from a copy of the original graph
    G = nx.Graph(original_graph)
    # re-wire the endpoints of edge-vertices
    for n, d in original_graph.nodes_iter(data=True):
        if d.get('edge', False) == True:
            # extract the endpoints
            endpoints = [u for u in original_graph.neighbors(n)]
            # assert (len(endpoints) == 2), 'ERROR: more than 2 endpoints'
            if len(endpoints) != 2:
                continue
            u = endpoints[0]
            v = endpoints[1]
            # add the corresponding edge
            G.add_edge(u, v, d)
            # remove the edge-vertex
            G.remove_node(n)
        if d.get('node', False) == True:
            # remove stale information
            G.node[n].pop('remote_neighbours', None)
    return G