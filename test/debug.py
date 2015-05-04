import sys
import os
os.nice(20)
sys.path.append("..")
from eden.converter.graph.gspan import gspan_to_eden
import graphlearn.graphlearn as gl
import itertools


def test_sampler():
    steps=100
    graphcount=40
    sampler=gl.GraphLearnSampler()
    sampler.load('../example/tmp/demo.ge')
    graphs = gspan_to_eden( '../example/bursi.pos.gspan' )
    graphs = itertools.islice(graphs,graphcount)

    # we test multicore and single:
    graphs = sampler.sample(graphs,
                            same_radius=False,
                            same_core_size=False,
                            sampling_interval=9999,
                            batch_size=int(graphcount/4)+1,
                            n_steps=steps,
                            n_jobs=4,
                            annealing_factor=0.9)
    #graphs = sampler.sample(graphs,same_radius=True,sampling_interval=9999,batch_size=2,n_steps=steps,n_jobs=4)
    for e in graphs:
        print e[0]


def test_fit():
    gr = gspan_to_eden( '../example/bursi.pos.gspan' )
    #radius_list=[2,4]
    #thickness_list=[2]

    #gr=itertools.islice(gr,50)

    sampler=gl.GraphLearnSampler()
    sampler.fit(gr,n_jobs=-1)
    sampler.save('../example/tmp/demo.ge')
    #graphlearn_utils.draw_grammar(sampler.local_substitutable_graph_grammar,5)
    print 'fitting done'




#test_fit()
test_sampler()


