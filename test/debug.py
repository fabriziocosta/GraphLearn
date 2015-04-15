
import sys
sys.path.append("..")


from adaptiveMHgraphsampler import adaptiveMHgraphsampler
import itertools

from eden.converter.graph.gspan import gspan_to_eden
graphs = gspan_to_eden( '../data/bursi.pos.gspan' )



nj=1
batch=160
steps=100
graphcount=160


imprules= {'n_jobs':nj , 'batch_size':batch,'improvement_steps':steps}
print imprules,graphcount

graphs= itertools.islice(graphs,graphcount)
sampler=adaptiveMHgraphsampler()
sampler.load('../data/demo.ge')
graphs = sampler.mass_improve_random(graphs,improvement_rules=imprules)
history=[]
for graphs_ in graphs:
    history.append(graphs_[-1].scorehistory)


