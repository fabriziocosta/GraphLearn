def sample_step(object, transformer, grammar, scorer, chooser):
    """
    Parameters
    ----------
    object:
    transformer: transform.transform(object) -> graph
    grammar: trained on graphs, lsgg.propose(graph) -> objects
    scorer: score.decision_function(object) -> number

    Returns
    -------
        object
    """
    graph = transformer.encode(object)
    proposals = transformer.decode( grammar.propose(graph) )
    scores = scorer.decision_function(proposals) # works on object
    object, scorer = chooser.choose(proposals, scores)
    return object, scorer


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
    import graphlearn as gl
    from graphlearn.score import SimpleDistanceEstimator as SDE
    import networkx as nx

    class transformer:
        def encode(self, thing):
            return thing
        def decode(self, thing):
            return thing

    class chooser:
        def choose(self, proposals,scores):
            return max(zip(proposals, scores), key=lambda x: x[1])

    lsgg = gl.test_get_grammar()
    graph = gl._edenize_for_testing(nx.path_graph(4))
    graph.node[3]['label'] = '5'
    score_estimator = SDE().fit(gl._edenize_for_testing(nx.path_graph(4)))
    graph,score= sample(graph,transformer(),lsgg,score_estimator,chooser(),n_steps=2,return_score=True)

    assert (0.000001 > abs(0.319274373045 - score))
