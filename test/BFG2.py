import sys
sys.path.append("..")
import matplotlib
matplotlib.use('Agg')

from eden.converter.graph.gspan import gspan_to_eden
from graphlearn import GraphLearnSampler
import itertools
from eden.graph import Vectorizer
import matplotlib.pyplot as plt

import utils.myeden as me

#2401 positives and 1936 negatives
radius=[2,4]
thickness=[2,4]
sampler =GraphLearnSampler(radius_list=radius,thickness_list=thickness)

graphs_pos= gspan_to_eden('../grammar/bursi.pos.gspan')
graphs_neg= gspan_to_eden('../grammar/bursi.neg.gspan')
testpos= gspan_to_eden('../grammar/bursi.pos.gspan')
testneg= gspan_to_eden('../grammar/bursi.neg.gspan')

vect=Vectorizer(complexity = 3)



NUMPOS=2401
NUMNEG=1936

#NUMPOS=27
#NUMNEG=27

testneg=itertools.islice(testneg,int(NUMNEG*.7)+1,NUMNEG)
testpos=itertools.islice(testpos,int(NUMPOS*.7)+1,NUMPOS)
testpos=vect.transform(testpos)
testneg=vect.transform(testneg)



def train_estimator_and_evaluate_testsets(pos_real,neg_real,pos_augmented,neg_augmented,test_pos,test_neg):
    testcount= float(NUMNEG*.3+NUMPOS*.3)
    vectorizer=Vectorizer(complexity = 3)
    pr= vectorizer.transform(pos_real)
    nr= vectorizer.transform(neg_real)

    pa=vectorizer.transform(pos_augmented)
    na=vectorizer.transform(neg_augmented)
    real_esti= me.graphlearn_fit_estimator( pr,nr )
    aug_esti= me.graphlearn_fit_estimator( pa,na )
    
    ori=0
    
    #print real_esti.predict(test_pos)
    for e in real_esti.predict(test_pos):
        if e == 1:
            ori+=1
    for e in real_esti.predict(test_neg):
        if e!=1:
            ori+=1
    imp=0
    for e in aug_esti.predict(test_pos):
        if e == 1:
            imp+=1
    for e in aug_esti.predict(test_neg):
        if e!=1:
            imp+=1        
    
    return imp/testcount,ori/testcount

def unpack(graphs):
    for graphlist in graphs:
        yield graphlist[0]

#generate datapoints: 
lenpo=int(NUMPOS*.7)
lenne=int(NUMNEG*.7)
originals=[]
improved=[]
percentages=[.2,.4,.6,.8,1]
percentages=[.2]
for perc in percentages:

    count_pos = int(lenpo*perc)
    count_neg = int(lenne*perc)

    graphs_pos, graphs_pos_ = itertools.tee(graphs_pos)
    graphs_neg, graphs_neg_ = itertools.tee(graphs_neg)



    pos = me.select_random(graphs_pos_, lenpo,count_pos )
    neg = me.select_random(graphs_neg_, lenne,count_neg )

    neg_,neg,neg__,neg___=itertools.tee(neg,4)
    pos_,pos,pos__,pos___=itertools.tee(pos,4)

    print 'negative sampler,',

    imprules= {'n_jobs':4 , 'batch_size':(count_neg/4)+1,'improvement_steps':50}
    sampler.fit(neg__)
    improved_neg = unpack (sampler.sample(neg___,improvement_rules=imprules)  )

    print 'positive sampler,',
    imprules= {'n_jobs':4 , 'batch_size':(count_pos/4)+1,'improvement_steps':50}
    sampler.fit(pos__)
    improved_pos = unpack(  sampler.sample(pos___,improvement_rules=imprules) )

    #testneg,testneg_=itertools.tee(testneg)
    #testpos,testpos_=itertools.tee(testpos)
    print 'evaluating..'
    imp,ori=train_estimator_and_evaluate_testsets( pos,neg, 
        itertools.chain(pos_,improved_pos),
        itertools.chain(neg_,improved_neg),
        testpos,testneg)
    improved.append(imp)
    originals.append(ori)
    print "done:"+str(perc)
    print "*"*80

print improved
print originals
# draw 
t = range(len(percentages))
plt.plot(t,originals ,'bs')
plt.plot(t, improved ,'g^')
plt.savefig('zomg2.png')
