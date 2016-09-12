import eden

# this is copy pasted in EVERY notebook. now its here
import matplotlib.pyplot as plt
import numpy as np


def plot_scores(scoreslist='list of lists',x=[], labels=[]):

    if len(labels)==0:
        labels= ['index: %d' % i for i in len(scoreslist)]
    if x==[]:
        x=range( len(scoreslist[0] ) )

    plt.figure(figsize=(10,5))


    for j,scores in enumerate(scoreslist):
        plt.plot(x,scores, label=labels[j])

    maa= max([ max(l) for l in scoreslist  ])
    mii= min([min(l) for l in scoreslist])
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlim(-.5, max(x) * 1.1)
    plt.ylim(mii*.9,maa*1.1)

    plt.show()



# eden vectorizer wants labels everywhere so we set them.
def edenize_graph(g):
    for a, b, d in g.edges(data=True):
        if 'label' not in d:
            d['label'] = ''
    for n,d in g.nodes(data=True):
        if 'label' not in d:
            d['label'] = str(n)
    return g



# use eden.grouper(inputs, 50)




# apply to all nodes in a graph
def node_operation(graph, f):
    # applies function to n,d of nodes(data=True)
    # if you want to do assignments do this: def my(n,d): d[dasd]=asd
    # if you want the result you may use lambda :)
    res = []
    for n, d in graph.nodes(data=True):
        res.append(f(n, d))
    return res

def map_node_operation(graphs,f):
    return map( lambda g:node_operation(g,f)  ,graphs)


def hash_eden_vector(vec):
    return hash(tuple(vec.data + vec.indices))


def remove_eden_annotation(graph):
    # eden contaminates graphs with all sorts of stuff..
    for attribute in ['weight','hlabel']:
        node_operation(graph, lambda n, d: d.pop(attribute, None))
    eden.graph._clean_graph(graph)
    return graph






def unique_graphs(graphs, vectorizer):
    # returns datamatrix, subgraphs
    map(remove_eden_annotation, graphs)
    data = vectorizer.transform(graphs)
    # remove duplicates   from data and subgraph_list
    data, indices = unique_csr(data)
    graphs = [graphs[i] for i in indices]
    return data, graphs


def delete_rows_csr(mat, indices, keep=False):
    '''
    Parameters
    ----------
    mat   csr matrix
    indices  list of indices to work on
    keep  should i delete or keep the indices

    Returns
    -------
        csr matrix
    '''
    indices = list(indices)
    if keep == False:
        mask = np.ones(mat.shape[0], dtype=bool)
    else:
        mask = np.zeros(mat.shape[0], dtype=bool)
    mask[indices] = keep
    return mat[mask]


def unique_csr(csr):
    # returns unique csr and a list of used indices
    unique = {hash_eden_vector(row): ith for ith, row in enumerate(csr)}
    indices = [ith for hashvalue, ith in unique.items()]
    indices.sort()
    return delete_rows_csr(csr, indices, keep=True), indices