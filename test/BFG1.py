import sys
sys.path.append("..")
import matplotlib
matplotlib.use('Agg')

from eden.converter.graph.gspan import gspan_to_eden
from adaptiveMHgraphsampler import adaptiveMHgraphsampler
from eden.graph import Vectorizer
import matplotlib.pyplot as plt
import itertools

from utils import myeden



sampler2 =adaptiveMHgraphsampler()
vectorizer=Vectorizer( complexity=3 )




def load_estimator(): 
    sampler2.load('../data/estimator')
    return sampler2.estimator


def make_estimator():

    neg = gspan_to_eden('../data/bursi.neg.gspan')
    pos= gspan_to_eden('../data/bursi.pos.gspan')
    pos = vectorizer.transform( pos )
    neg = vectorizer.transform( neg )
    sampler2.estimator= myeden.my_fit_estimator(positive_data_matrix=pos, negative_data_matrix=neg)
    sampler2.save('../data/estimator')

    
estimator=make_estimator()



def unpack(graphs):
    for graphlist in graphs:
        yield graphlist[-1]

def doit():

    sampler = adaptiveMHgraphsampler()
    graphs_pos= gspan_to_eden('../data/bursi.pos.gspan')
    #generate datapoints:
    lenpo=int(2401*.3)
    originals=[]
    improved=[]
    radius=[2,4]
    thickness=[2,4]
    percentages=[.01, .05, .12, .25, .5 ,1 ]
    percentages=[.1]
    for perc in percentages:
        count = lenpo*perc


        graphs_pos, graphs_pos_ = itertools.tee(graphs_pos)
        graphs_pos_ = myeden.select_random(graphs_pos_, lenpo,count )
        graphs_pos_ = itertools.islice(graphs_pos_, count )
        graphs_pos_,graphs_pos__,graphs_pos___ = itertools.tee(graphs_pos_,3)
        
        sampler.train_estimator_and_extract_grammar(graphs_pos__,radius,thickness,n_jobs=4)
        imprules= {'n_jobs':4 , 'batch_size':(count/4)+1,'improvement_steps':50}
        improved_graphs = sampler.mass_improve_random(graphs_pos_,improvement_rules=imprules)
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
