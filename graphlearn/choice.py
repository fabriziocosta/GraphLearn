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

class SelectProbN(object):

    def __init__(self,n):
        self.n=n

    def select(self, proposals,scores):
        neg = min(scores)
        scores = [s-neg for s in scores]
        stuff = list (zip(proposals,scores))
        return  list(zip(*random.choices(stuff,scores,k=self.n)))



def test_SelectMaxN():
    s=SelectMaxN(2)
    assert s.select(['A',"B","C","D"],[.5,.4,.3,.6]) == [('D', 'A'), (0.6, 0.5)] 
       


