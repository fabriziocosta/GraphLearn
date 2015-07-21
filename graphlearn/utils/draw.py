import pylab as plt
from eden.util.display import draw_graph, draw_graph_set
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit
from collections import defaultdict
from graphlearn.utils import calc_stats_from_grammar
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


def graph_clean(graph):
    '''
    in the precess of creating a new graph,
    we marked the nodes that were used as interface and core.
    here we remove the marks.

    also this is a copy of the function with the same name in graphtools.
    it exists twice because python complained about cyclic imports
    :param graph:
    :return:
    '''
    for n, d in graph.nodes(data=True):
        d.pop('core', None)
        d.pop('interface', None)
        d.pop('root', None)


def plot_charts(data1, data2=None, xlabel=None, ylabel=None, size=(10, 4), log_scale=True):
    plt.figure(figsize=size)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data1, color='blue', lw=2)
    plt.plot(data1,
             linestyle='None',
             markerfacecolor='white',
             markeredgecolor='blue',
             marker='o',
             markeredgewidth=2,
             markersize=8)
    if data2 is not None:
        plt.plot(data2, color='red', lw=2)
        plt.plot(data2,
                 linestyle='None',
                 markerfacecolor='white',
                 markeredgecolor='red',
                 marker='o',
                 markeredgewidth=2,
                 markersize=8)
    if log_scale:
        plt.yscale('log')
    plt.xlim(-0.2, len(data1) + 0.2)
    plt.ylim(0.8)
    plt.show()


def draw_grammar_stats(grammar, size=(10, 4)):
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
    # a=[ (c[k],cc[k]) for k in c.keys()]
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
    nc = [v for v in c.values()]
    nc.sort()
    d = defaultdict(int)
    for e in nc:
        d[e] += 1
    dp = []
    for i in range(max(nc)):
        if i in d:
            dp.append(d[i])
        else:
            dp.append(d[i])

    plot_charts(dp, size=size)


def draw_center(graph, root_node, radius):
    dist = nx.single_source_shortest_path_length(graph, root_node, radius)
    graph.node[root_node]['color'] = 0.5
    draw_graph(nx.Graph(graph.subgraph(dist)), edge_label=None, vertex_color='color')


def set_ids(graph):
    for n, d in graph.nodes_iter(data=True):
        d['id'] = str(n)


def display(graph,
            size=6,
            font_size=15,
            node_size=200,
            node_border=False,
            show_direction=False,
            edge_color=None,
            contract=False,
            vertex_color='color',
            vertex_label='label',
            edge_label=None,
            **args):

    if show_direction:
        contract = False
    if contract:
        graph = contract_edges(graph)
    graph2 = graph.copy()
    set_colors(graph2)

    if show_direction:
        for n, d in graph2.nodes(data=True):
            if 'edge' in d:
                ne = graph2.neighnors(n)
                for e in ne:
                    graph2[n][e]['color'] = 'red'
    if vertex_label == 'id':
        set_ids(graph2)

    draw_graph(graph2,
               size=size,
               node_size=node_size,
               node_border=node_border,
               font_size=font_size,
               edge_color=edge_color,
               vertex_color=vertex_color,
               vertex_label=vertex_label,
               edge_label=edge_label,
               **args)


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
            graph_clean(g)
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


def draw_grammar(grammar, n_productions=None, n_graphs_per_line=5, size=4, **args):
    if n_productions is None:
        n_productions = len(grammar)

    if len(grammar) < n_productions:
        n_productions = len(grammar)

    for i in range(n_productions):
        interface = grammar.keys()[i]
        core_cid_dict = grammar[interface]

        cips = [core_cid_dict[chash] for chash in core_cid_dict.keys()]

        for cip in cips:
            cip.graph.graph['frequency'] = ' frequency:%s' % cip.count
        graphs = [cip.graph for cip in cips]

        # dists = [core_cid_dict[chash].distance_dict for i, chash in enumerate(core_cid_dict.keys()) \
        # if i < 5]
        print 'interface: ' + str(interface)
        freq = lambda graph: graph.graph['frequency']
        draw_graph_set_graphlearn(graphs,
                                  n_graphs_per_line=n_graphs_per_line,
                                  size=size,
                                  headlinehook=freq,
                                  **args)


def get_score_of_graph(graph):
    return "%s%s" % (' score: ', str(graph.graph.get('score', '?')))


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


def draw_graph_set_graphlearn(graphs, n_graphs_per_line=5, size=4, contract=True, vertex_color=None, **args):
    graphs = list(graphs)

    if contract:
        graphs = [contract_edges(g) for g in graphs]

    if vertex_color is None:
        for g in graphs:
            set_colors(g)
        vertex_color = 'col'

    # for e in graphs:
    #    e.graph['info']= get_score_of_graph(e)

    draw_graph_set(graphs, n_graphs_per_line=n_graphs_per_line, size=size, vertex_color=vertex_color, **args)


def contract_edges(original_graph):
    """
        stealing from eden...
        because i draw cores and interfaces there may be edge-nodes
        that have no partner, eden gives error in this case.
        i still want to see them :)
    """
    # start from a copy of the original graph
    graph = nx.Graph(original_graph)
    # re-wire the endpoints of edge-vertices
    for n, d in original_graph.nodes_iter(data=True):
        if d.get('edge', False) is True:
            # extract the endpoints
            endpoints = [u for u in original_graph.neighbors(n)]
            # assert (len(endpoints) == 2), 'ERROR: more than 2 endpoints'
            if len(endpoints) != 2:
                continue
            u = endpoints[0]
            v = endpoints[1]
            # add the corresponding edge
            graph.add_edge(u, v, d)
            # remove the edge-vertex
            graph.remove_node(n)
        if d.get('node', False) is True:
            # remove stale information
            graph.node[n].pop('remote_neighbours', None)
    return graph


def draw_learning_curve(data_first=None,
                        data_second=None,
                        measure=None,
                        x_axis=None,
                        delta=0.1,
                        scaling=100,
                        fname=None):
    """
    Accepts as input an iterator over lists of numbers.
    Draws the exponential decay grpah over the means of lists.
    """

    def learning_curve_function(x, a, b):
        return a * (1 - np.exp(-b * x))

    x_axis = np.array(x_axis)
    mean_originals = []
    for originals in data_first:
        mean_originals.append(np.mean(np.array(originals)))

    mean_originals_and_samples = []
    for originals_and_samples in data_second:
        mean_originals_and_samples.append(np.mean(np.array(originals_and_samples)))

    a, b = curve_fit(learning_curve_function, x_axis, mean_originals)
    c, d = curve_fit(learning_curve_function, x_axis, mean_originals_and_samples)

    x_axis_fit = np.linspace(x_axis.min(), x_axis.max(), 100)
    mean_originals_fit = learning_curve_function(x_axis_fit, *a)
    mean_originals_and_samples_fit = learning_curve_function(x_axis_fit, *c)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('Exponential Decay Learning Curves')
    # plt.subplots_adjust(left=0.04, right=0.35, top=0.9, bottom=0.25)

    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_title('Learning Curve Comparison for %s' % measure)
    ax1.set_xlabel('Dataset Percentage Used for Training')
    ax1.set_ylabel('%s Value' % measure)

    plt.boxplot(data_first, positions=(x_axis + delta) * scaling, notch=False)
    plt.plot((x_axis + delta) * scaling, mean_originals, 'ro', label='')
    plt.plot((x_axis_fit) * scaling, mean_originals_fit, 'r-', label='Original')

    plt.box_axisplot(data_second, positions=(x_axis - delta) * scaling, notch=False)
    plt.plot((x_axis - delta) * scaling, mean_originals_and_samples, 'go', label='')
    plt.plot((x_axis_fit) * scaling, mean_originals_and_samples_fit, 'g-', label='Original+sampled')
    plt.grid()
    plt.legend(loc='lower right')
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
