
import sys
sys.path.append("..")


from graphlearn import adaptiveMHgraphsampler
import itertools

from eden.converter.graph.gspan import gspan_to_eden
graphs = gspan_to_eden( '../data/bursi.pos.gspan' )



nj=0
batch=160
steps=20
graphcount=20


imprules= {'n_jobs':nj , 'batch_size':batch,'improvement_steps':steps}
print imprules,graphcount
sampler=adaptiveMHgraphsampler()


sampler.load('../data/demo.ge')
#sampler.train_estimator_and_extract_grammar(graphs,[2,4],[2],n_jobs=4)
#sampler.save('../data/demo.ge')



graphs= itertools.islice(graphs,graphcount)
graphs = sampler.sample_set(graphs,improvement_rules=imprules)
history=[]
for graphs_ in graphs:
    history.append(graphs_[0].score_history)


for l in history:
    print l