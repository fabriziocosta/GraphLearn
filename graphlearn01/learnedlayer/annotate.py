from eden.graph import Vectorizer
from graphlearn01 import utils
from sklearn.linear_model import SGDClassifier
from graphlearn01.estimate import ExperimentalOneClassEstimator
import eden
import multiprocessing as mp
import numpy as np
import sklearn

class twoclass(SGDClassifier):
    # THE HACK IS NOW GETTING EVEN MORE EVIL
    def __init__(self):
        self.clazz= SGDClassifier(loss='log')

    def fit(self,X,y, crossval=False):

        if crossval:
            print "layers crossvalscore:",sklearn.model_selection.cross_val_score(SGDClassifier(loss='log'),X, y).mean()

        self.clazz.fit(X,y)
        self.intercept_= self.clazz.intercept_
        self.classes_= self.clazz.classes_
        return self

    # eden cant annotate two classes if the esti is not a sgdregressor
    #  -> this hack is made!
    '''
    details: decission function returns a one d array.
    eden only accepts these if the estimater is instance of sgdregressor.
    so i make a two d array from my 1 d array.
    if i hack something like this in the future maybe the intercept array needs to be provided..
    (see the annotator code)
    '''

    # default guy:
    #def decision_function(self, vector):
    #    answer =  super(self.__class__,self).decision_function(vector)
    #    return np.vstack((answer, (answer-1))).T

    def decision_function(self,vector):
        return self.clazz.predict_proba(vector)

    '''
    def decision_function(self, vector):
        #answer =  super(self.__class__,self).decision_function(vector)
        answer =  super(self.__class__,self).predict_proba(vector)
        return np.vstack((answer, (answer-1))).T
    '''


class Annotator():

    def __init__(self, multiprocess=True, score_attribute='importance',vectorizer=eden.graph.Vectorizer(),debug=False, annotate_dilude_scores=False):
        self.score_attribute=score_attribute
        self.vectorizer=vectorizer
        self.multi_process=multiprocess
        self.trained=False
        self.debug=debug
        self.annotate_dilude_scores = annotate_dilude_scores

    def fit(self, graphs_pos, graphs_neg=[]):

        map(utils.remove_eden_annotation,graphs_pos+graphs_neg)
        map(lambda x: utils.node_operation(x, lambda n,d: d.pop('importance',None)), graphs_pos+graphs_neg)
        map( lambda graph: graph.graph.pop('mass_annotate_mp_was_here',None) ,graphs_pos+graphs_neg)

        if graphs_neg:
            #print 'choosing to train binary esti'
            self.estimator = twoclass() #SGDClassifier()
            classes= [1]*len(graphs_pos)+[-1]*len(graphs_neg)

            self.estimator.fit(self.vectorizer.transform(graphs_pos+graphs_neg),classes,crossval=self.debug)
        else:
            print 'annotator fits 1 class esti..'
            self.estimator = ExperimentalOneClassEstimator()
            self.estimator.fit(self.vectorizer.transform(graphs_pos))
        return self


    def fit_transform(self,graphs_p, graphs_n=[]):
        self.fit(graphs_p,graphs_n)
        return self.transform(graphs_p+graphs_n)

    def transform(self,graphs):
        return  self.annotate(graphs)

    def annotate(self,graphs):
        if not graphs:
            return []


        res =  mass_annotate_mp(graphs,self.vectorizer,score_attribute=self.score_attribute,estimator=self.estimator,
                                multi_process=self.multi_process, annotate_dilude_scores=self.annotate_dilude_scores)

        if False: # checking for annotation bug
            for g in res:
                for n,d in g.nodes(data=True):
                    if type(d[self.score_attribute])==type(None):
                        from graphlearn01.utils import draw
                        print "graphlearn01 learnedlayer annotate.py"
                        draw.graphlearn(g)

        return res


def mass_annotate_mp(inputs, vectorizer, score_attribute='importance', estimator=None, multi_process=False, annotate_dilude_scores=False):
    '''
    graph annotation is slow. i dont want to do it twice in fit and predict :)
    '''

    #  1st check if already annotated
    #if inputs[0].graph.get('mass_annotate_mp_was_here', False):
    #    return inputs

    if multi_process == False:
        inputs = filter(lambda v: v is not None, inputs)


        res = list(vectorizer.annotate(inputs, estimator=estimator))

        def dilute_graph( graph):
            for n, d in graph.nodes(data=True):
                neighsum = [graph.node[other][score_attribute][0] for other in graph.neighbors(n)]
                if neighsum != []:
                    allfacs = neighsum + [graph.node[n][score_attribute][0]] * len(neighsum)
                    score = sum(allfacs) / float(len(allfacs))
                else:
                    score = d[score_attribute][0]
                d['tmpscore'] = score

            for n, d in graph.nodes(data=True):
                d[score_attribute] = [d['tmpscore'], 0]
                # self.attribute =  lambda x: x['tmpscore']
        if annotate_dilude_scores:
            map(dilute_graph,res)

        #if invert_score:
        #    def f(n,d): d['importance'] = -d['importance']
        #    res=utils.map_node_operation(res,f)

        res[0].graph['mass_annotate_mp_was_here'] = True
        return res
    else:
        pool = mp.Pool()
        mpres = [eden.apply_async(pool, mass_annotate_mp, args=(graphs, vectorizer, score_attribute, estimator)) for
                 graphs in eden.grouper(inputs, 50)]
        result = []
        for res in mpres:
            result += res.get()
        pool.close()
        pool.join()
        return result
