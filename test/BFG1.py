import sys
sys.path.append("..")
import matplotlib
matplotlib.use('Agg')

from eden.converter.graph.gspan import gspan_to_eden
from graphlearn.graphlearn import GraphLearnSampler
from eden.graph import Vectorizer
import matplotlib.pyplot as plt
import itertools
from graphlearn.utils import myeden
from eden.util import fit_estimator as eden_fit_estimator
from eden.util import selection_iterator as picker
from sklearn.linear_model import SGDClassifier
import random


# a vectorizer
vectorizer = Vectorizer( complexity=3 )

# select 1st element in an iterator
def unpack(graphs):
    for graphlist in graphs:
        yield graphlist[0]


def make_estimator(pos,neg):
    pos = vectorizer.transform( pos )
    neg = vectorizer.transform( neg )
    esti = eden_fit_estimator(SGDClassifier(), positive_data_matrix=pos,
                                        negative_data_matrix=neg)
    return esti



# positive set contains 2401 elements, of which we use 30% to test of we cen improve them ,
# the rest is used for the oracle
lenpo=int(2401*.3)


# we select those 30% randomly:
splitset= range(2014)
random.shuffle(splitset)
sample=splitset[:lenpo]
oracle=splitset[lenpo:]

path='../example/'

# we create an oracle
estimator=make_estimator(picker(gspan_to_eden(path+'bursi.pos.gspan'),oracle),gspan_to_eden(path+'bursi.neg.gspan'))
print 'estimator ok'



# ok we create an iterator over the graphs we want to work with...
graphs_pos= picker( gspan_to_eden(path+'bursi.pos.gspan') , sample)


# save results here:
originals=[]
improved=[]



# we want to use an increasing part of the test set..
percentages=[.1, .2, .4, .6, .8 ,1 ]

sampler = GraphLearnSampler()

for perc in percentages:

    # we work with count many graphs
    count = int(lenpo*perc)
    # make copy of graphiterator
    # select count random elements
    # triplicate  the count long iterator
    graphs_pos, graphs_pos_ = itertools.tee(graphs_pos)
    x=range(count)
    random.shuffle(x)
    graphs_pos_ = picker(graphs_pos_, x )
    graphs_pos_,graphs_pos__,graphs_pos___ = itertools.tee(graphs_pos_,3)


    # do sampling
    sampler.fit(graphs_pos__, grammar_n_jobs=4)

    improved_graphs = sampler.sample(graphs_pos_,
                                     same_radius=False,
                                     max_size_diff=True,
                                     sampling_interval=9999,
                                     select_cip_max_tries=100,
                                     batch_size=int(count/4)+1,
                                     n_steps=100,
                                     n_jobs=-1,
                                     improving_threshold=0.9)



    #calculate the score of the improved versions
    #calculate score of the originals
    avg_imp=sum( [estimator.decision_function(e) for e in vectorizer.transform(unpack(improved_graphs)) ] )/count
    avg_ori=sum( [estimator.decision_function(e) for e in vectorizer.transform(graphs_pos___)] )/count
    improved.append(avg_imp)
    originals.append(avg_ori)


t = range(len(percentages))
# originals are blue
# improved ones are green

print originals
print improved
plt.plot(t,originals ,'bs')
plt.plot(t, improved ,'g^')
plt.savefig('zomg.png')

