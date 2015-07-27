import sys
import os
os.nice(20)
sys.path.append("..")
from eden.converter.graph.gspan import gspan_to_eden
import graphlearn.graphlearn as gl
import itertools
from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=1)

def test_sampler():
    steps=20
    graphcount=8
    sampler=gl.GraphLearnSampler()
    sampler.load('../example/tmp/demo_50.ge')
    graphs = gspan_to_eden( '../example/bursi.pos.gspan' )
    graphs = itertools.islice(graphs,graphcount)

    # we test multicore and single:
    graphs = sampler.sample(graphs,
                            same_radius=False,
                            max_core_size_diff=False,
                            sampling_interval=9999,
                            batch_size=2,
                            probabilistic_core_choice=True,
                            n_steps=steps,
                            n_jobs=1,
                            improving_threshold=0.9,
                            keep_duplicates=False)
    #graphs = sampler.sample(graphs,same_radius=True,sampling_interval=9999,batch_size=2,n_steps=steps,n_jobs=4)
    for e in graphs:
        print e


def test_fit():
    gr = gspan_to_eden( '../example/bursi.pos.gspan' )
    #radius_list=[2,4]
    #thickness_list=[2]

    gr=itertools.islice(gr,50)

    sampler=gl.GraphLearnSampler()
    sampler.fit(gr,n_jobs=-1)
    sampler.save('../example/tmp/demo_50.ge')
    #graphlearn_utils.draw_grammar(sampler.local_substitutable_graph_grammar,5)
    print 'fitting done'




#test_fit()
test_sampler()


