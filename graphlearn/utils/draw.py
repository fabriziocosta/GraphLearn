'''
make graphics of any kind ( mostly graphs)
'''

import pylab as plt
from eden.display import draw_graph_set
from eden.display import draw_graph as eden_draw_graph

import networkx as nx
import numpy as np
from scipy.optimize import curve_fit
from collections import defaultdict
import logging
import copy

logger = logging.getLogger(__name__)


def calc_stats_from_grammar(grammar):
    count_corehashes = defaultdict(int)
    count_interfacehashes = defaultdict(int)
    corecounter = defaultdict(int)
    intercounter = defaultdict(int)
    for ih in grammar.keys():
        for ch in grammar[ih].keys():
            # go over all the combos
            count_corehashes[ch] += 1
            count_interfacehashes[ih] += 1
            count = grammar[ih][ch].count
            corecounter[ch] += count
            intercounter[ih] += count
    return count_corehashes, count_interfacehashes, corecounter, intercounter


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
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    plt.grid()
    ax = plt.subplot(111)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    data1_x = [3 * x - 1 for x in range(len(data1))]
    ax.bar(data1_x, data1, width=1.5, color='r')
    if data2 is not None:
        data2_x = [3 * x - 2 for x in range(len(data2))]
        ax.bar(data2_x, data2, width=1.5, color='b')
    if log_scale:
        plt.yscale('log')
    plt.xlim(-1, len(data1) + 1)
    plt.ylim(0.1)
    plt.show()


def plot_charts2(data1, data2=None, xlabel=None, ylabel=None, size=(10, 4), log_scale=True):
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
    print "how often do we see interface hashes"
    a = [(i[k], ii[k]) for k in i.keys()]
    a.sort()
    a0 = [e[0] for e in a]
    a1 = [e[1] for e in a]
    print '# productions: %d' % sum(a0)

    print 'x = # interfaces (total: %d)' % len(i)
    print 'y=number of cores(ihash), y=sum Of Core Counts(ihash)'
    plot_charts(a0, a1, xlabel='# interfaces', ylabel='counts', size=size)

    print 'how often was this corehash seen?'
    a = [(c[k], cc[k]) for k in c.keys()]
    a.sort()
    a0 = [e[0] for e in a]
    a1 = [e[1] for e in a]
    print 'x = # cores (total: %d)' % len(c)
    print 'y = in Y interfaces(chash), y= sum Of Count Over All Interfaces(chash)'
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


def graphlearn_layered1(graphs, **args):
    '''
    if there is a graph whose nodes have a layers annotation, use this
    Args:
        graphs:
        **args:

    Returns:
    '''

    def calc_avg_position(nodelist, posdict):
        # print 'calc avg pos'
        if len(nodelist) == 0:
            print 'bad node list'
            return (0, 0)
        xpos = sum([posdict[i][0] for i in nodelist]) / len(nodelist)
        ypos = sum([posdict[i][1] for i in nodelist]) / len(nodelist)
        return (xpos, ypos)

    def get_leafes(graph, node):
        # print 'get leafes %d' % node
        if graph.node[node].get('contracted', 0) == 0:
            return [node]
        nodes = [node]
        leafes = []
        while len(nodes) > 0:
            current = nodes.pop()
            # contraction also includes edge nodes -> ignore those
            if current not in graph.nodes():
                continue
            if 'contracted' in graph.node[current]:
                children = list(graph.node[current]['contracted'])
                nodes += children
            else:
                leafes.append(current)

        # print leafes
        return leafes

    finished_graphs = []
    poslist = []
    for graph in graphs:
        # how many layers are there? also make a list of nodes for each layer
        nodelayer = defaultdict(list)
        layercount = -1
        for n, d in graph.nodes(data=True):
            layer = d.get('layer', -1)
            layercount = max(layercount, layer)
            nodelayer[d['layer']].append(n)
        if layercount == -1:
            print "layer annotation missing in graph"
            break

        # layout layer 0
        pos = nx.graphviz_layout(graph.subgraph(nodelayer[0]), prog='neato', args="-Gmode=KK")

        # pos attribute loks like this:
        # pos = {i: (rna_object.get(i).X, rna_object.get(i).Y)
        #           for i in range(len(graph.graph['structure']))}



        for layerid in range(1, layercount + 1):

            new_positions = {}
            # nodes in the layer:
            nodes = nodelayer[layerid]
            for node in nodes:
                nulllayernodes = get_leafes(graph, node)
                new_positions[node] = calc_avg_position(nulllayernodes, pos)

            # move all the nodes by such and such
            # nodes in prev layer:
            moveby_x = max(pos[i][0] for i in nodelayer[layerid - 1]) + 100
            # moveby_y = max( pos[i][1] for i in nodelayer[layerid-1] ) - 30
            moveby_y = ((-1) ** layerid) * 30
            for k, v in new_positions.items():
                new_positions[k] = (v[0] + moveby_x, v[1] + moveby_y)

            pos.update(new_positions)

        # color dark edged:
        if False:
            for node, d in graph.nodes(data=True):
                if 'contracted' in d:
                    for other in d['contracted']:
                        if other in graph.nodes():
                            graph[node][other]['dark_edge_color'] = d['layer']

        finished_graphs.append(graph)
        poslist.append(pos)

    # draw
    args['size_x_to_y_ratio'] = layercount + 1
    args['pos'] = poslist
    args['dark_edge_color'] = 'dark_edge_color'
    graphlearn(finished_graphs, **args)


def graphlearn_layered2(graphs, **args):
    '''
    this is to draw a graph that has its layers as graph.graph['origial']

    Args:
        graphs:
        **args:

    Returns:

    '''

    DEBUG = False

    def calc_avg_position(nodelist, posdict):
        # print 'calc avg pos'
        if len(nodelist) == 0:
            print 'bad node list'
            return (0, 0)
        xpos = sum([posdict[i][0] for i in nodelist]) / len(nodelist)
        ypos = sum([posdict[i][1] for i in nodelist]) / len(nodelist)
        return (xpos, ypos)

    finished_graphs = []
    poslist = []

    for graph in graphs:

        # make a list of all the graphs
        layered_graphs = [graph]
        while 'original' in graph.graph:
            layered_graphs.append(graph.graph['original'])
            graph = graph.graph['original']
        maxlayers = len(layered_graphs)

        # make the layout for the biggest one :)
        pos = nx.graphviz_layout(layered_graphs[-1], prog='neato', args="-Gmode=KK")
        if DEBUG: print 'biggest:', pos

        # pos attribute loks like this:
        # pos = {i: (rna_object.get(i).X, rna_object.get(i).Y)
        #           for i in range(len(graph.graph['structure']))}

        for i in range(len(layered_graphs) - 2, -1, -1):
            new_positions = {}
            for node in layered_graphs[i].nodes():
                new_positions[node] = calc_avg_position(layered_graphs[i].node[node].get('contracted', set()), pos)
            if DEBUG: print 'new posis', new_positions
            # move all the nodes by such and such
            # nodes in prev layer:
            minpos = min([pos[n][0] for n in layered_graphs[i + 1].nodes()])
            moveby_x = max([pos[n][0] for n in layered_graphs[i + 1].nodes()]) + 200 - minpos
            #print moveby_x
            moveby_y = ((-1) ** i) * 30
            for k, v in new_positions.items():
                new_positions[k] = (v[0] + moveby_x, v[1] + moveby_y)

            if DEBUG: print 'new posis updated', new_positions
            pos.update(new_positions)

        g = nx.union_all(layered_graphs)
        for n, d in g.nodes(data=True):
            for n2 in d.get('contracted', []):
                g.add_edge(n, n2, nesting=True, label='')
        finished_graphs.append(g)
        poslist.append(pos)

    # draw
    args['size_x_to_y_ratio'] = maxlayers
    args['pos'] = poslist
    args['dark_edge_color'] = 'dark_edge_color'
    graphlearn(finished_graphs, **args)


def graphlearn_dict(dict, **args):
    # idea is that the values are graphs
    for k, l in dict.items():
        print k
        graphlearn(l[:5], **args)


def draw_center(graph, root_node, radius, **args):
    dist = nx.single_source_shortest_path_length(graph, root_node, radius)
    graph.node[root_node]['color'] = 0.5
    graphlearn(nx.Graph(graph.subgraph(dist)), edge_label=None, vertex_color='color', **args)


def set_ids(graph):
    for n, d in graph.nodes_iter(data=True):
        d['id_LABEL'] = str(n)


def graphlearn(graphs,
               size=6,
               font_size=15,
               # node_size=200,
               # node_border=False,
               show_direction=False,
               abstract_color=None,
               edge_color=None,
               contract=False,
               vertex_color=None,
               vertex_label='label',
               edge_label=None,
               edge_alpha=.5,
               **args):
    if isinstance(graphs, nx.Graph):
        graphs = [graphs]

    graphs = copy.deepcopy(graphs)

    for graph in graphs:
        if show_direction:
            contract = False

        if vertex_color is None:
            set_colors(graph)

        # if vertex_color_attribute='importance':
        #    set_colors_importance(graph)

        if show_direction:
            for n, d in graph.nodes(data=True):
                if 'edge' in d:
                    ne = graph.neighbors(n)
                    for e in ne:
                        graph[n][e]['color'] = 1
        if abstract_color != None:
            for a, b, d in graph.edges(data=True):
                if 'contracted' in graph.node[a] and 'contracted' in graph.node[b]:
                    d['color'] = abstract_color
                else:
                    d['color'] = 'gray'

        if vertex_label == 'id' or args.get("secondary_vertex_label", "no") == 'id':
            set_ids(graph)

        if vertex_label == 'importance' or args.get('secondary_vertex_label', '') == 'importance':
            for n, d in graph.nodes(data=True):
                d['importance'] = round(d['importance'], 2)

    # now we need to change the attribute
    # because there is a label collission in json graph saving
    if vertex_label == 'id':
        vertex_label = 'id_LABEL'

    if args.get("secondary_vertex_label", "no") == 'id':
        args["secondary_vertex_label"] = 'id_LABEL'

    if vertex_label == 'importance' and vertex_color == None:
        vertex_color = 'importance'
        args['colormap'] = 'inferno'

    if vertex_color is None:
        vertex_color = 'col'

    if show_direction or abstract_color:
        edge_color = 'color'
        edge_alpha = 1.0

    if contract:
        # tmp=[]
        # for graph in graphs:
        #    tmp.append(  contract_edges(graph) )
        # graphs=tmp
        graphs = [contract_edges(g) for g in graphs]

    draw_graph_set(graphs,
                   size=size,
                   # node_size=node_size,
                   # node_border=node_border,
                   font_size=font_size,
                   edge_color=edge_color,
                   vertex_color=vertex_color,
                   vertex_label=vertex_label,
                   edge_label=edge_label,
                   edge_alpha=edge_alpha,
                   **args)


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


def cip_to_drawable_graph(cips=[], graphs=[], mark_root=False):
    regraphs = []
    if not graphs:
        for cip in cips:
            graph = cip.graph
            if mark_root:
                graph.node[cip.distance_dict[0][0]]['root'] = True
                graph.node[cip.distance_dict[0][0]].pop('core')
            regraphs.append(graph)
    else:
        for c, g in zip(cips, graphs):
            remove_colors(g)
            graph_clean(g)
            g2 = g.copy()
            d = {0: 'core'}
            if mark_root:
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


def draw_grammar(grammar,
                 n_productions=None,
                 n_graphs_per_line=5,
                 n_graphs_per_production=10,
                 size=4,
                 abstract_interface=False,
                 title_key='frequency',
                 **args):
    if abstract_interface:
        n_graphs_per_production -= 1

    if n_productions is None:
        n_productions = len(grammar)

    if len(grammar) < n_productions:
        n_productions = len(grammar)

    most_prolific_productions = sorted(
        [(len(grammar[interface]), interface) for interface in grammar],
        reverse=True)

    for i in range(n_productions):
        interface = most_prolific_productions[i][1]
        # interface = grammar.keys()[i]
        core_cid_dict = grammar[interface]

        cips = [core_cid_dict[chash] for chash in core_cid_dict.keys()]

        for cip in cips:
            cip.graph.graph['frequency'] = ' frequency:%s' % cip.count

        most_frequent_cips = sorted([(cip.count, cip) for cip in cips], reverse=True)
        graphs = [cip.graph for count, cip in most_frequent_cips]
        # graphs =[cip.abstract_view for count, cip in most_frequent_cips]


        graphs = graphs[:n_graphs_per_production]
        # dists = [core_cid_dict[chash].distance_dict for i, chash in enumerate(core_cid_dict.keys()) \
        # if i < 5]
        print('interface id: %s [%d options]' % (interface, len(grammar[interface])))

        # uebersampler gets to do this because he is the uebersampler
        if abstract_interface:
            if 'abstract_view' in cips[0].__dict__:
                graphs = [most_frequent_cips[0][1].abstract_view] + graphs

        graphlearn(graphs,
                   n_graphs_per_line=n_graphs_per_line,
                   size=size,
                   title_key=title_key,
                   **args)


def remove_colors(g, key='col'):
    for n, d in g.nodes(data=True):
        d[key] = 'white'


def contract_edges(original_graph):
    """
        stealing from eden...
        because i draw cores and interfaces there may be edge-nodes
        that have no partner, eden gives error in this case.
        i still want to see them :)
    """
    # start from 0a copy of the original graph
    # graph = nx.Graph(original_graph)

    graph = original_graph.copy()
    # re-wire the endpoints of edge-vertices
    for n, d in original_graph.nodes_iter(data=True):
        if d.get('edge', False) is True:
            # extract the endpoints
            endpoints = [u for u in original_graph.neighbors(n)]
            if len(endpoints) == 2:
                u = endpoints[0]
                v = endpoints[1]
            elif len(endpoints) == 1:  # support for digraph
                u = endpoints[0]
                v = original_graph.predecessors(n)[0]
            else:
                print "DRAW:SOMETHING IS WRONG IN CONTRACT EDGES  ends:", len(endpoints), n
                continue

            # add the corresponding edge
            nd = {}
            # ATTENTION
            # i update the edge first, so that d can overwrite an eventual existing label attribute
            # also i think the info from the node (d) is most important and should no be overwritten by the edge (og[n][u])
            nd.update(original_graph[n][u])
            nd.update(d)

            # print nd,d,original_graph[n][u]
            graph.add_edge(v, u, nd)
            # remove the edge-vertex
            graph.remove_node(n)

        if d.get('node', False) is True:
            # remove stale information
            # note to self: since i imported this from eden, i am not sure what this does  currently
            graph.node[n].pop('remote_neighbours', None)
    return nx.Graph(graph)


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

    plt.boxplot(data_second, positions=(x_axis - delta) * scaling, notch=False)
    plt.plot((x_axis - delta) * scaling, mean_originals_and_samples, 'go', label='')
    plt.plot((x_axis_fit) * scaling, mean_originals_and_samples_fit, 'g-', label='Original+sampled')
    plt.grid()
    plt.legend(loc='lower right')
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
