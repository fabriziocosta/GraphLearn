import logging
logger = logging.getLogger(__name__)

# calculates all the neighbors of a given graph
# graphlearn_examples/more_examples/neighbors.py


def interface_to_cips(input_cip, grammar):
    if input_cip.interface_hash in grammar.productions:
        for cip in grammar.productions[input_cip.interface_hash].values():
            if cip.core_hash != input_cip.core_hash:
                yield cip


def decomposer_neighbors(decomposer, grammar):

    # self.decomposer.make_new_decomposer(self.graph_transformer.re_transform_single(new_graph))
    # the above is how gl usually obtains a new deomp..
    # just fyi ...

    orig_cips = decomposer.all_core_interface_pairs(
        radius_list=grammar.radius_list,
        thickness_list=grammar.thickness_list)

    for orig_cip in [a for asd in orig_cips for a in asd]:
        for cip in interface_to_cips(orig_cip, grammar):
            yield decomposer.core_substitution(orig_cip.graph, cip.graph)


def graph_neighbors(decomposer, grammar, graph_transformer):
    orig_cips = decomposer.all_core_interface_pairs(
        radius_list=grammar.radius_list,
        thickness_list=grammar.thickness_list)

    for orig_cip in [a for asd in orig_cips for a in asd]:
        for cip in interface_to_cips(orig_cip, grammar):
            g = decomposer.core_substitution(orig_cip.graph, cip.graph)
            try:
                g = decomposer.make_new_decomposer(
                    graph_transformer.re_transform_single(g))
            except:
                continue
            yield g
