import math
import networkx as nx

#########
# set print symbol
#########

colordict = {'black': 0, 'red': 1,
             'green': 2,
             'yellow': 3,
             'blue': 4,
             'cyan': 6,
             'magenta': 5,
             'gray': 7}


def color(symbol, col='red'):
    '''http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python'''
    return '\x1b[1;3%d;48m%s\x1b[0m' % (colordict[col], symbol)


def defaultcolor(d):
    if 'core' in d:
        return 'cyan'
    elif 'interface' in d:
        return 'magenta'
    elif 'edge' in d:
        return 'blue'
    else:
        return 'red'


def set_print_symbol(g, bw=False, label='label', colorlabel=None):
    for n, d in g.nodes(data=True):
        symbol = d['label']
        if bw:
            d['asciisymbol'] = symbol

        elif colorlabel:
            d['asciisymbol'] = color(symbol, d[colorlabel])
        else:
            d['asciisymbol'] = color(symbol, defaultcolor(d))
    return g


####
# coordinate setter
###

def nx_to_ascii(graph,
                size=10,
                debug=None,
                bw=False):
    ymax = size
    xmax = ymax * 2

    '''
        debug would be a path to the folder where we write the dot file.
    '''

    # ok need coordinates and a canvas
    canvas = [list(' ' * (xmax + 1)) for i in range(ymax + 1)]
    pos = nx.graphviz_layout(graph, prog='neato', args="-Gratio='2'")

    # alternative way to get pos. for chem graphs.
    # seems a little bit unnecessary now... maybe ill built it later
    # import molecule
    # chem=molecule.nx_to_rdkit(graph)
    # m.GetConformer().GetAtomPosition(0)

    # transform coordinates
    weird_maxx = max([x for (x, y) in pos.values()])
    weird_minx = min([x for (x, y) in pos.values()])
    weird_maxy = max([y for (x, y) in pos.values()])
    weird_miny = min([y for (x, y) in pos.values()])

    xfac = (weird_maxx - weird_minx) / xmax
    yfac = (weird_maxy - weird_miny) / ymax
    for key in pos.keys():
        wx, wy = pos[key]
        pos[key] = (int((wx - weird_minx) / xfac), int((wy - weird_miny) / yfac))

    # draw nodes
    for n, d in graph.nodes(data=True):
        if 'label' in d:
            symbol = str(d['label'])
        else:
            symbol = str(n)

        x, y = pos[n]
        for e in symbol:

            canvas[y][x] = d['asciisymbol']
            if x < xmax:
                x += 1
            else:
                continue

    # draw edges
    for (a, b) in graph.edges():
        ax, ay = pos[a]
        bx, by = pos[b]
        resolution = max(3, int(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)))
        dx = float((bx - ax)) / resolution
        dy = float((by - ay)) / resolution
        for step in range(resolution):
            x = int(ax + dx * step)
            y = int(ay + dy * step)
            if canvas[y][x] == ' ':
                canvas[y][x] = "." if bw else color('.', col='black')

    canvas = '\n'.join([''.join(e) for e in canvas])
    if debug:
        path = "%s/%s.dot" % (debug, hash(graph))
        canvas += "\nwriting graph:%s" % path
        nx.write_dot(graph, path)

    return canvas


######
# contract and horizontalize
######

def contract_graph(graph):
    import eden.graph as eg
    graph = eg._revert_edge_to_vertex_transform(graph)
    return graph


def transpose(things):
    return map(list, zip(*things))


def makerows(graph_canvazes):

    g = map(lambda x: x.split("\n"), graph_canvazes)
    g = transpose(g)
    res = ''
    for row in g:
        res += "".join(row) + '\n'
    return res

#######
# main printers
#######


def make_picture(g, bw=False, colorlabel=None, contract=False, label='label', size=10, debug=None):
    if type(g) != list:
        g = [g]

    if contract:
        g = map(contract_graph, g)

    g = map(lambda x: set_print_symbol(x, bw=bw, label=label, colorlabel=colorlabel), g)

    g = map(lambda x: nx_to_ascii(x, size=size, debug=debug, bw=bw), g)
    return makerows(g)


def gprint(g, **kwargs):
    print make_picture(g, **kwargs)


# test
if __name__ == "__main__":
    graph = nx.path_graph(3)
    stuff = nx_to_ascii(graph)
    print stuff
