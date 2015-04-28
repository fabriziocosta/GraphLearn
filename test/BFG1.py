import sys
sys.path.append("..")
import matplotlib
matplotlib.use('Agg')

from eden.converter.graph.gspan import gspan_to_eden
from graphlearn import GraphLearnSampler
from eden.graph import Vectorizer
import matplotlib.pyplot as plt
import itertools
from utils import myeden



sampler2 =GraphLearnSampler()
vectorizer=Vectorizer( complexity=3 )




def load_estimator(): 
    sampler2.load('../grammar/estimator')
    return sampler2.estimator


def make_estimator():

    neg = gspan_to_eden('../grammar/bursi.neg.gspan')
    pos= gspan_to_eden('../grammar/bursi.pos.gspan')
    pos = vectorizer.transform( pos )
    neg = vectorizer.transform( neg )
    sampler2.estimator= myeden.graphlearn_fit_estimator(positive_data_matrix=pos, negative_data_matrix=neg)
    sampler2.save('../grammar/estimator')

    
estimator=load_estimator()
print 'estimator ok'

def unpack(graphs):
    for graphlist in graphs:
        yield graphlist[0]

def doit():
    radius=[2,4]
    thickness=[2,4]
    sampler = GraphLearnSampler(radius_list=radius,thickness_list=thickness)
    graphs_pos= gspan_to_eden('../grammar/bursi.pos.gspan')
    #generate datapoints:
    lenpo=int(2401*.3)
    originals=[]
    improved=[]

    percentages=[.01, .05, .12, .25, .5 ,1 ]
    percentages=[.1]
    for perc in percentages:
        count = int(lenpo*perc)


        graphs_pos, graphs_pos_ = itertools.tee(graphs_pos)
        graphs_pos_ = myeden.select_random(graphs_pos_, lenpo,count )
        graphs_pos_ = itertools.islice(graphs_pos_, count )
        graphs_pos_,graphs_pos__,graphs_pos___ = itertools.tee(graphs_pos_,3)
        
        sampler.fit(graphs_pos__,n_jobs=4)
        imprules= {'n_jobs':4 , 'batch_size':(count/4)+1,'improvement_steps':50}
        improved_graphs = sampler.sample(graphs_pos_,improvement_rules=imprules)
        avg_imp=sum( [estimator.decision_function(e) for e in vectorizer.transform(unpack(improved_graphs),n_jobs=4) ] )/count
        avg_ori=sum( [estimator.decision_function(e) for e in vectorizer.transform(graphs_pos___,n_jobs=4)] )/count
        
        improved.append(avg_imp)
        originals.append(avg_ori)

        

    t = range(len(percentages))
    plt.plot(t,originals ,'bs')
    plt.plot(t, improved ,'g^')
    plt.savefig('zomg.png')
    print originals
    print improved

doit()
