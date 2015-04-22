
import sys
sys.path.append("..")


from graphlearn import GraphLearnSampler
import itertools

from eden.converter.graph.gspan import gspan_to_eden




def test_sampler():
    nj=0
    batch=160
    steps=20
    graphcount=20

    graphs = gspan_to_eden( '../data/bursi.pos.gspan' )


    imprules= {'n_jobs':nj , 'batch_size':batch,'improvement_steps':steps}
    print imprules,graphcount
    sampler=GraphLearnSampler()
    sampler.load('../data/demo.ge')
    #sampler.train_estimator_and_extract_grammar(graphs,[2,4],[2],n_jobs=4)
    #sampler.save('../data/demo.ge')

    graphs= itertools.islice(graphs,graphcount)
    graphs = sampler.sample(graphs,improvement_rules=imprules)
    history=[]
    for graphs_ in graphs:
        history.append(graphs_[0].score_history)
    for l in history:
        print l




import graphlearn as gl
import itertools

def test_fit():
    gr = gspan_to_eden( '../data/bursi.pos.gspan' )
    radius_list=[2,4]
    thickness_list=[2]

    gr=itertools.islice(gr,100)
    sampler=gl.GraphLearnSampler(radius_list,thickness_list)
    sampler.fit(gr,n_jobs=4)


    sampler.save('../data/demo.ge')
    #myutils.draw_grammar(sampler.substitute_grammar,5)


test_fit()