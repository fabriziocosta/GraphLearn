
# calculates all the neighbors of a given graph 
# graphlearn_examples/more_examples/neighbors.py 


import graphlearn.decompose 

def suitors(cip,grammar):
    if cip.interface_hash in grammar.productions:
        for other in grammar.productions[cip.interface_hash].values():
            if other.core_hash != cip.core_hash:
                yield other
    

def neighbors(graph, grammar):
    decomp = decompose.Decomposer(data=graph)
    return getallneighbors(decomp,grammar)


def getallneighbors(decomposer, grammar): 



    
    #self.decomposer.make_new_decomposer(self.graph_transformer.re_transform_single(new_graph))
    # the above is how gl usually obtains a new deomp.. 
    # just fyi ... 

    orig_cips = decomposer.all_core_interface_pairs(
            radius_list = grammar.radius_list,
            thickness_list=grammar.thickness_list)

    for orig_cip in [a for asd in orig_cips for a in asd]:
        for new_cip in suitors(orig_cip,grammar):
                yield decomposer.core_substitution(orig_cip.graph, new_cip.graph)


