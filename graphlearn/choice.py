import random


class SelectMax:
    def select(self, proposals,scores):
        return max(zip(proposals, scores), key=lambda x: x[1])


class SelectMaxN(object):
    def __init__(self,n):
        self.n=n

    def select(self, proposals,scores):
        instances = list(zip(proposals, scores))
        instances.sort(key=lambda x: x[1],reverse=True)
        return list(zip(*[ instances[x] for x in range(self.n) ]))


class SelectClassic(object):
    def __init__(self,reg=0.0):
        self.reg = reg

    def select(self, proposals, scores):
        # assuming ill get 2 proposals.. 0 is the new one.
        #'''accept new graph if it is better'''
        
        new = (proposals[0], scores[0])
        old = (proposals[1], scores[1])
        if scores[0]<=0 or scores[1]<=0: 
            return new if scores[0]<scores[1] else old 
        if scores[0] > scores[1]:
            return new
        elif random.random() < ((scores[0]/scores[1])  -self.reg):
            return new
        return old

       

class SelectProbN(object):

    def __init__(self,n):
        self.n=n

    def select(self, proposals,scores):
        neg = min(scores)
        scores = [s-neg for s in scores]
        stuff = list (zip(proposals,scores))
        if self.n > 1:
            return  list(zip(*random.choices(stuff,scores,k=self.n)))
        else:
            return  random.choices(stuff,scores,k=1)[0]



def test_SelectMaxN():
    s=SelectMaxN(2)
    assert s.select(['A',"B","C","D"],[.5,.4,.3,.6]) == [('D', 'A'), (0.6, 0.5)] 
       


