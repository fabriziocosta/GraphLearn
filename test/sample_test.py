



from graphlearn.estimators import simple_directed_estimator as sde


import graphlearn.sample as sample
import lsgg_test
import networkx as nx


def test_sampler():
    lsggg=lsgg_test.get_grammar()
    graph = lsgg_test.edenize(nx.path_graph(4))
    graph.node[3]['label']='5'
    score_estimator= sde( lsgg_test.edenize(nx.path_graph(4)) )

    sampler = sample.Sampler(grammar=lsggg,score_estimator=score_estimator, n_steps=2)


    for i in range(2):
        graph, score = sampler.transform(graph).next()
    assert (0.000001 > abs( 0.319274373045 - score))


if __name__=="__main__":
    test_sampler()
