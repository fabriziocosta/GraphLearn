from graphlearn3.util import util
def sample_step(object, transformer, grammar, scorer, chooser):
    """
    Parameters
    ----------
    object:
    transformer: test.test(object) -> graph
    grammar: trained on graphs, lsgg.propose(graph) -> objects
    scorer: score.decision_function(object) -> number

    Returns
    -------
        object
    """
    graph = transformer.encode(object)
    util.valid_gl_graph(graph)
    proposal_graphs = grammar.propose(graph)

    proposal_objects = transformer.decode(proposal_graphs)
    scores = scorer.decision_function(proposal_objects)
    object, score = chooser.choose(proposal_objects, scores)

    return object, score


def sample(graph, transformer=None, grammar=None, scorer=None, chooser=None, n_steps=10, return_score=False):
    for i in range(n_steps):
        graph, score = sample_step(graph, transformer, grammar, scorer,chooser)
    if return_score:
        return graph, score
    return graph


def fit():
    pass

def optimize():
    pass




def test_sample_step():
    from graphlearn3.score import SimpleDistanceEstimator as SDE
    import networkx as nx
    from graphlearn3 import choose
    import graphlearn3.test.transformutil as transformutil

    lsgg = util.test_get_grammar()
    graph = util._edenize_for_testing(nx.path_graph(4))
    graph.node[3]['label'] = '5'
    score_estimator = SDE().fit(util._edenize_for_testing(nx.path_graph(4)))
    graph,score= sample(graph, transformutil.no_transform(), lsgg, score_estimator, choose.Chooser(), n_steps=2, return_score=True)

    assert (0.000001 > abs(0.319274373045 - score)), score
    print("sambledestdone")
