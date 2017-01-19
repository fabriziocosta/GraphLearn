from eden.graph import Vectorizer
from graphlearn import utils
from sklearn.linear_model import SGDClassifier
from graphlearn.estimate import ExperimentalOneClassEstimator
import eden
import multiprocessing as mp



class Annotator():

    def __init__(self, multiprocess=True, score_attribute='importance'):
        self.score_attribute=score_attribute
        self.vectorizer=Vectorizer()
        self.multi_process=multiprocess
        self.trained=False

    def fit(self, graphs_pos, graphs_neg=[]):

        if self.trained:
            return self
        self.trained=True
        map(utils.remove_eden_annotation,graphs_pos+graphs_neg)
        map(lambda x: utils.node_operation(x, lambda n,d: d.pop('importance',None)), graphs_pos+graphs_neg)
        map( lambda graph: graph.graph.pop('mass_annotate_mp_was_here',None) ,graphs_pos+graphs_neg)

        if graphs_neg:
            #print 'choosing to train binary esti'
            self.estimator = SGDClassifier()
            classes= [1]*len(graphs_pos)+[-1]*len(graphs_neg)
            self.estimator.fit(self.vectorizer.transform(graphs_pos+graphs_neg),classes)
        else:
            self.estimator = ExperimentalOneClassEstimator()
            self.estimator.fit(self.vectorizer.transform(graphs_pos))
        return self


    def fit_transform(self,graphs_p, graphs_n=[]):
        self.fit(graphs_p,graphs_n)
        return self.transform(graphs_p),self.transform(graphs_n)

    def transform(self,graphs):
        return  self.annotate(graphs)

    def annotate(self,graphs,neg=False):
        if not graphs:
            return []
        return mass_annotate_mp(graphs,self.vectorizer,score_attribute=self.score_attribute,estimator=self.estimator,
                                multi_process=self.multi_process, invert_score=neg)



def mass_annotate_mp(inputs, vectorizer, score_attribute='importance', estimator=None, multi_process=False, invert_score=False):
    '''
    graph annotation is slow. i dont want to do it twice in fit and predict :)
    '''
    #  1st check if already annotated
    if inputs[0].graph.get('mass_annotate_mp_was_here', False):
        return inputs

    if multi_process == False:
        inputs = filter(lambda v: v is not None, inputs)
        res = list(vectorizer.annotate(inputs, estimator=estimator))
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
