import sys
sys.path.append('..')
import os
os.nice(19)

import graphlearn.utils.draw as myutils
import graphlearn.directedsampler as cl
from eden.converter.graph.gspan import gspan_to_eden
import itertools
import matplotlib.pyplot as plt


steps=500

sampler=cl.directedSampler()

sampler.fit(gspan_to_eden( 'bursi.pos.gspan' ),  n_jobs=4)
sampler.save('tmp/cluster.ge')
exit()
#sampler.load('tmp/cluster.ge')



graphs = gspan_to_eden( 'bursi.pos.gspan' )
#graphs = itertools.islice(graphs,9)
graphs = sampler.sample(graphs,
                        sampling_interval=int(steps/3)+1,
                        batch_size=1,
                        n_steps=steps,
                        n_jobs=1,
                        select_cip_max_tries = 200,
                        accept_annealing_factor= 2,
                        doXgraphs= 9
                        )


history=[]
for  i, (result,info) in enumerate(graphs):
    history.append(info['score_history'])
    #myutils.draw_many_graphs(info['graphs'])


'''
t = range(steps+1) 
for h in history[:3]:
    plt.plot(t, h)
plt.show()
t = range(steps+1) 
for h in history[3:6]:
    plt.plot(t, h)
plt.show()
t = range(steps+1) 
for h in history[6:]:
    plt.plot(t, h)
plt.show()
'''

