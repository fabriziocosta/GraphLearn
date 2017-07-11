'''
allws interactive sampling.
works with the associated notebook.
'''
from graphlearn01.decompose import Decomposer
import graphlearn01


class mywrap(Decomposer):
    def clean(self):
        return

    def real_clean(self):
        graphlearn01.decompose.graph_clean(self._base_graph)

    def make_new_decomposer(self, transformout):
        return mywrap(transformout,node_entity_check=self.node_entity_check,nbit=self.nbit)
# this file is here to hide the uglyness from the notebooks
# i should use the init_only flag in the sampler for initialisation.

def setparameters(sampler):
    sampler.step = 3  # whatever
    sampler._init_grammar_prep()


# for the easy mode:
def easy_get_new_graphs(graphwrap, sampler):
    res = []
    graphwrap.clean()
    res = [sampler._propose(graphwrap) for i in range(8)]

    for i, gw in enumerate(res):
        gw._base_graph.graph['info'] = str(i)
    # for i in range(8):
    #    gr2=      nx.Graph(gr)
    #    cip =     sampler.select_original_cip(gr2)
    #    newcip =  sampler._select_cips(cip).next()
    #    newgr=    graphtools.core_substitution(gr2, cip.graph, newcip.graph)
    #    res.append(newgr)

    return res


# used in non easy mode

def getargz(sampler):
    return {'radius_list': sampler.lsgg.radius_list,
            'thickness_list': sampler.lsgg.thickness_list,

            # 'filter':sampler.node_entity_check
            }

import draw
def get_cips(graphman, sampler, root, d, debug=False):
    cips = graphman.rooted_core_interface_pairs(root, **d)
    if debug:
        draw.graphlearn([c.graph for c in cips])

    res = []
    counter = 0
    for cip in cips:
        if cip.interface_hash in sampler.lsgg.productions:
            new_cips = sampler.lsgg.productions[cip.interface_hash].values()
            for nc in new_cips:
                # save the original_cip_graph for replacement later
                nc.orig = cip.graph
                nc.graph.graph['info'] = str(counter)
                counter += 1
            res += new_cips

    return res
