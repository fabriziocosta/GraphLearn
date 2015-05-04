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

from sklearn.linear_model import SGDClassifier
sampler2 = GraphLearnSampler()
vectorizer = Vectorizer( complexity=3 )



def make_estimator():
    neg = gspan_to_eden('../example/bursi.neg.gspan')
    pos= gspan_to_eden('../example/bursi.pos.gspan')
    pos = vectorizer.transform( pos )
    neg = vectorizer.transform( neg )
    esti = eden_fit_estimator(SGDClassifier(), positive_data_matrix=pos,
                                        negative_data_matrix=neg)
    return esti



estimator=make_estimator()
print 'estimator ok'

def unpack(graphs):
    for graphlist in graphs:
        yield graphlist[0]

def doit():
    sampler = GraphLearnSampler()
    graphs_pos= gspan_to_eden('../example/bursi.pos.gspan')
    #generate datapoints:
    lenpo=int(2401*.3)


    originals=[]
    improved=[]

    percentages=[.01, .05, .12, .25, .5 ,1 ]
    percentages=[.1]
    for perc in percentages:
        count = int(lenpo*perc)

        # make copy of graphiterator
        # select count random elements
        # triplicate  the count long iterator
        graphs_pos, graphs_pos_ = itertools.tee(graphs_pos)
        graphs_pos_ = myeden.select_random(graphs_pos_, lenpo,count )
        graphs_pos_,graphs_pos__,graphs_pos___ = itertools.tee(graphs_pos_,3)


        # do sampling
        sampler.fit(graphs_pos__,n_jobs=4)

        improved_graphs = sampler.sample( graphs_pos_,
                            same_radius=False,
                            same_core_size=False,
                            sampling_interval=9999,
                            batch_size=int(count/4)+1,
                            n_steps=200,
                            n_jobs=4,
                            annealing_factor=0.9)

        #calculate the score of the improved versions
        #calculate score of the originals
        avg_imp=sum( [estimator.decision_function(e) for e in vectorizer.transform(unpack(improved_graphs),n_jobs=4) ] )/count
        avg_ori=sum( [estimator.decision_function(e) for e in vectorizer.transform(graphs_pos___,n_jobs=4)] )/count
        improved.append(avg_imp)
        originals.append(avg_ori)


    t = range(len(percentages))
    # originals are blue
    # improved ones are green
    plt.plot(t,originals ,'bs')
    plt.plot(t, improved ,'g^')
    plt.savefig('zomg.png')
    print originals
    print improved

doit()
