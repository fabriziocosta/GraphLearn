

from rdkit import Chem
from graphlearn.utils.draw_openbabel import graph_to_molfile
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from IPython.core.display import  display
import networkx as nx

def nx_to_chem(nx):
    molstring = graph_to_molfile(nx)
    return  Chem.MolFromMolBlock(molstring, sanitize=False)

def set_coordinates(chemlist):
    for m in chemlist:
        tmp = AllChem.Compute2DCoords(m)

def draw(graphs, n_graphs_per_line=5, size=250, title_key=None, titles=None):

    # we want a list of graphs
    if isinstance(graphs, nx.Graph):
        graphs = [graphs]

    # make molecule objects
    chem=[  nx_to_chem(graph) for graph in graphs]
    # calculate coordinates:
    set_coordinates(chem)

    # take care of the subtitle of each graph
    if title_key:
        legend=[g.graph[title_key] for g in graphs]
    elif titles:
        legend=titles
    else:
        legend=[str(i) for i in range(5)]

    # make the image
    image= Draw.MolsToGridImage(chem, molsPerRow=n_graphs_per_line, subImgSize=(size, size), legends=legend)

    # display on the spot
    display( image )



def sdf_to_nx(file):
    suppl = Chem.SDMolSupplier(file)
    # this is given in the example, not sure if its a list or an iterator.. want list:)
    #for mol in suppl:
    #    print(mol.GetNumAtoms())
    for mol in suppl:
        yield sdMol_to_nx(mol)
    #return [ for mol in suppl]


def sdMol_to_nx(mol):
    #print dir(chem)
    #print chem.GetNumAtoms()
    graph = nx.Graph()
    for e in mol.GetAtoms():
        graph.add_node(e.GetIdx(),label=e.GetSymbol())
    for b in mol.GetBonds():
        graph.add_edge(b.GetBeginAtomIdx(),b.GetEndAtomIdx(), label=str(int(b.GetBondTypeAsDouble())))
    return graph




