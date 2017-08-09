import eden
from eden.graph import _label_preprocessing

import matplotlib.pyplot as plt
import numpy as np
import graphlearn01.decompose as decompose

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


def make_bar_plot(labels=('G1', 'G2', 'G3', 'G4', 'G5'), means=(20, 35, 30, 35, 27), stds=(2, 3, 4, 1, 2)):
    N = len(labels)
    ind = np.arange(N)
    width = .5  # 0.35
    plt.figure(figsize=(14, 5))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(18)

    p1 = plt.bar(ind, means, width, color='#69ACEA', yerr=stds, edgecolor='None', ecolor='#444444')
    plt.axhline(y=38, color='black', linewidth=3)
    plt.ylabel("log odds score", fontsize=20)
    plt.xlabel("number of graphs", fontsize=20)
    # plt.xlabel("number of graphs",fontsize=20)
    plt.title('Scores by training size', fontsize=20)
    plt.xticks(ind + width / 2, labels)
    plt.yticks(np.arange(0, 100, 10))
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
    for attribute in ['weight','vector','features', 'importance']:
        node_operation(graph, lambda n, d: d.pop(attribute, None))
    eden.graph._clean_graph(graph)
    return graph


def unique_graphs(graphs, vectorizer):
    # returns datamatrix, subgraphs
    graphs=map(remove_eden_annotation, graphs)
    data = vectorizer.transform(graphs)
    # remove duplicates   from data and subgraph_list
    data, indices = unique_csr(data)
    graphs = [graphs[i] for i in indices]
    return data, graphs


def unique_graphs_graphlearn_graphhash(graphs):
    map (_label_preprocessing,graphs)
    hashes = map(lambda x:decompose.graph_hash(x,2**20-1,'hlabel'),graphs)
    di={h:i  for i,h in enumerate(hashes) }
    return [graphs[i] for i in di.values()]



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

'''
 catching errors and printing them
 except Exception as exc:
                print (exc)
                print (traceback.format_exc(10))
'''





''' ... dump graphs instead of drawing for future use...
import base64
import dill
def dump64(g,**argz):
    return base64.b64encode(dill.dumps(g))+"\n\n"
draw.graphlearn = lol
'''


import subprocess
def shexec(cmd):
    '''
    :param cmd:
    :return: (exit-code, stderr, stdout)

    the subprocess module is chogeum.. here is a workaround
    '''
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, stderr = process.communicate()
    retcode = process.poll()
    return (retcode,stderr,output)

