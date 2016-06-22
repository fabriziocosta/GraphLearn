'''
comfortably use openbabel:

draw for drawing graphs

graph_to_molfile() is closely related to a eden function that does
nearly the same but is broken for our purposes at least.
'''

import pybel
import networkx as nx
from IPython.display import display, HTML
import eden.graph as edengraphtools


def draw(graphs, n_graphs_per_line=5, size=200, d3=False, title_key=None):
    '''

    Parameters
    ----------
    graphs:  list of graphs
    n_graphs_per_line: int
        draw this many graphs in a line
    size: int
        size in the svg image, svg-default is 100 which is very small.

    d3: bool (false)
        will result in 3d images

    title_key: basestring, None
        graph.graph[titles_key] will be displayed as graph title


    Returns
    -------
        void
    '''
    # if graphs is only a single graph, we convert to a list
    if isinstance(graphs, nx.Graph):
        graphs = [graphs]

    # d3 is false for now.
    pybel.ipython_3d = d3

    while graphs:
        some_graphs = graphs[:n_graphs_per_line]

        # printing one line...
        headers = []
        graph_texts = []

        for graph in some_graphs:

            headers.append(graph.graph.get(title_key, ''))
            babelgraph = nx_to_pybel(graph)

            if d3 == False:
                svgtext = babelgraph._repr_svg_()
                svgtext = svgtext.split("\n")
                svg_header = svgtext[0]
                svg_header = svg_header.replace('height="100"', 'height="%d"' % size)
                svg_header = svg_header.replace('width="100"', 'width="%d"' % size)
                svgtext = [svg_header] + svgtext[1:]
                svgtext = '\n'.join(svgtext)
                graph_texts.append(svgtext)
            else:
                svgtext = babelgraph._repr_html_()
                graph_texts.append(svgtext)

        html = ['<table>']
        if title_key is not None:
            html += ["<th>{}</th>".format(h) for h in headers] + ["</tr><tr>"]

        # html=["<table><tr>"]
        for svg in graph_texts:
            html.append("<td>" + svg + "</td>")
        html.append("</tr></table>")
        display(HTML(''.join(html)))

        graphs = graphs[n_graphs_per_line:]


def graph_to_molfile(graph):
    '''
    Parameters
    ----------
    graph: nx.graph

    Returns
    -------
        sdf_string


    this is taken from eden.
    atom_line += d['label'].ljust(3) <- this is changed from eden.
    '''
    symbols = {'1': 'H',
               '2': 'He',
               '3': 'Li',
               '4': 'Be',
               '5': 'B',
               '6': 'C',
               '7': 'N',
               '8': 'O',
               '9': 'F',
               '10': 'Ne',
               '11': 'Na',
               '12': 'Mg',
               '13': 'Al',
               '14': 'Si',
               '15': 'P',
               '16': 'S',
               '17': 'Cl',
               '18': 'Ar',
               '19': 'K',
               '20': 'Ca',
               '21': 'Sc',
               '22': 'Ti',
               '23': 'V',
               '24': 'Cr',
               '25': 'Mn',
               '26': 'Fe',
               '27': 'Co',
               '28': 'Ni',
               '29': 'Cu',
               '30': 'Zn',
               '31': 'Ga',
               '32': 'Ge',
               '33': 'As',
               '34': 'Se',
               '35': 'Br',
               '36': 'Kr',
               '37': 'Rb',
               '38': 'Sr',
               '39': 'Y',
               '40': 'Zr',
               '41': 'Nb',
               '42': 'Mo',
               '43': 'Tc',
               '44': 'Ru',
               '45': 'Rh',
               '46': 'Pd',
               '47': 'Ag',
               '48': 'Cd',
               '49': 'In',
               '50': 'Sn',
               '51': 'Sb',
               '52': 'Te',
               '53': 'I',
               '54': 'Xe',
               '55': 'Cs',
               '56': 'Ba',
               '57': 'La',
               '58': 'Ce',
               '59': 'Pr',
               '60': 'Nd',
               '61': 'Pm',
               '62': 'Sm',
               '63': 'Eu',
               '64': 'Gd',
               '65': 'Tb',
               '66': 'Dy',
               '67': 'Ho',
               '68': 'Er',
               '69': 'Tm',
               '70': 'Yb',
               '71': 'Lu',
               '72': 'Hf',
               '73': 'Ta',
               '74': 'W',
               '75': 'Re',
               '76': 'Os',
               '77': 'Ir',
               '78': 'Pt',
               '79': 'Au',
               '80': 'Hg',
               '81': 'Tl',
               '82': 'Pb',
               '83': 'Bi',
               '84': 'Po',
               '85': 'At',
               '86': 'Rn',
               '87': 'Fr',
               '88': 'Ra',
               '89': 'Ac',
               '90': 'Th',
               '91': 'Pa',
               '92': 'U',
               '93': 'Np',
               '94': 'Pu',
               '95': 'Am',
               '96': 'Cm',
               '97': 'Bk',
               '98': 'Cf',
               '99': 'Es',
               '100': 'Fm',
               '101': 'Md',
               '102': 'No',
               '103': 'Lr',
               '104': 'Rf',
               '105': 'Db',
               '106': 'Sg',
               '107': 'Bh',
               '108': 'Hs',
               '109': 'Mt',
               '110': 'Ds',
               '111': 'Rg',
               '112': 'Uub',
               '113': 'Uut',
               '114': 'Uuq',
               '115': 'Uup',
               '116': 'Uuh',
               '117': 'Uus',
               '118': 'Uuo'}

    graph=edengraphtools._revert_edge_to_vertex_transform(graph)
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default', label_attribute=None)
    # creating an SDF file from graph:
    # The header block, i.e. the first three lines, may be empty:
    sdf_string = "Networkx graph to molfile\n\n\n"

    # After the header block comes the connection table:
    # First the counts line - step by step
    counts_line = ""
    # Number of atoms

    counts_line += str(len(graph.nodes())).rjust(3)
    # Number of bonds
    counts_line += str(len(graph.edges())).rjust(3)
    # Number of atom lists
    counts_line += '  0'
    # Three blank spaces, then the chirality flag
    counts_line += '     1'
    # Five identical blocks
    counts_line += '  0' * 5
    # Finish with 0999 V2000
    counts_line += '999 V2000\n'
    sdf_string += counts_line

    # Atom block - this contains one atom line per atom in the molecule
    for n, d in graph.nodes_iter(data=True):
        atom_line = ''
        # Set all coordinates to 0
        atom_line += '    0.0000    0.0000    0.0000 '
        # Atom symbol: it should be the entry from the periodic table, using
        # atom type for now
        # atom_line += symbols.get(d['discrete_label']).ljust(3)
        atom_line += d['label'].ljust(3)
        # Lots of unnecessary atomic information:
        atom_line += ' 0  0  0  0  0  0  0  0  0  0  0  0\n'
        sdf_string += atom_line

    # Bond block
    for i, j, k in graph.edges_iter(data=True):
        edge_line = ''
        # Use the stored atom ids for the bonds, plus one
        edge_line += str(i + 1).rjust(3) + \
                     str(j + 1).rjust(3) + k['label'].rjust(3)
        # More information
        edge_line += '  0  0  0  0\n'
        sdf_string += edge_line

    sdf_string += 'M  END'
    # sdf_string += 'M END\n\n$$$$'


    return sdf_string


def nx_to_pybel(graph):
    # from  eden.converter.molecule.obabel  import graph_to_molfile
    molstring = graph_to_molfile(graph)
    # thing=pybel.readstring('sdf',molstring)
    thing = pybel.readstring('mol', molstring)
    return thing
