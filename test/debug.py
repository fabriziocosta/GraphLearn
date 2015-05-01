
import sys
sys.path.append("..")
import itertools
from eden.converter.graph.gspan import gspan_to_eden

import graphlearn.graphlearn as gl
import itertools


def test_sampler():
    steps=20
    sampler=gl.GraphLearnSampler()
    sampler.load('../example/tmp/demo.ge')
    graphs = gspan_to_eden( '../example/bursi.pos.gspan' )
    graphs = itertools.islice(graphs,4)

    # we test multicore and single:
    graphs = sampler.sample(graphs,same_radius=False,same_core_size=True,sampling_interval=9999,batch_size=1,n_steps=steps,n_jobs=0)
    #graphs = sampler.sample(graphs,same_radius=True,sampling_interval=9999,batch_size=2,n_steps=steps,n_jobs=4)

    for e in graphs:
        print e


def test_fit():
    gr = gspan_to_eden( '../example/bursi.pos.gspan' )
    radius_list=[2,4]
    thickness_list=[2]
    gr=itertools.islice(gr,100)
    sampler=gl.GraphLearnSampler(radius_list,thickness_list)
    sampler.fit(gr,n_jobs=-1)
    sampler.save('../example/tmp/demo.ge')
    #graphlearn_utils.draw_grammar(sampler.local_substitutable_graph_grammar,5)
    print 'fitting done'

test_fit()

test_sampler()

