
import sys
sys.path.append("..")


from graphlearn import GraphLearnSampler
import itertools

from eden.converter.graph.gspan import gspan_to_eden



def test_sampler():
    steps=20
    sampler=gl.GraphLearnSampler()
    sampler.load('../data/demo.ge')
    graphs = gspan_to_eden( '../data/bursi.pos.gspan' )
    graphs = itertools.islice(graphs,4)



    # we test multicore and single:
    graphs = sampler.sample(graphs,same_radius=False,same_core_size=True,sampling_interval=9999,batch_size=1,n_steps=steps,n_jobs=0)

    #graphs = sampler.sample(graphs,same_radius=True,sampling_interval=9999,batch_size=2,n_steps=steps,n_jobs=4)


    history=[]

    for e in graphs:
        print e

    #for (result,info) in graphs:
    #    history.append(info['score_history'])
    #print  history





import graphlearn as gl
import itertools

def test_fit():
    gr = gspan_to_eden( '../data/bursi.pos.gspan' )
    radius_list=[2,4]
    thickness_list=[2]
    #gr=itertools.islice(gr,100)
    sampler=gl.GraphLearnSampler(radius_list,thickness_list)
    sampler.fit(gr,n_jobs=-1)
    sampler.save('../data/demo.ge')
    #graphlearn_utils.draw_grammar(sampler.local_substitutable_graph_grammar,5)


#test_fit()

test_sampler()

