'''
OneClassEstimator
'''
import numpy
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier, LinearRegression
import random
import sklearn
from utils import hash_eden_vector as hashvec
import numpy as np

# Train the model using the training sets



def simple_fit_estimator(esti, pos, neg):
    y=np.array([1]*pos.shape[0] + [-1]*neg.shape[0])
    matrix = vstack([pos,neg], format="csr")
    esti.fit(matrix,y)
    return esti



class Regressor:
    '''
    there might be a bug connected to nx.digraph..
    '''

    def __init__(self, regressor=LinearRegression()):
        '''
        Parameters
        ----------
        regressor: regressor

        Returns
        -------
        '''
        self.status = 'new'
        self.regressor = regressor
        self.inverse_prediction = False

    def fit(self, data_matrix, values, random_state=None):

        if random_state is not None:
            random.seed(random_state)
        # use eden to fitoooOoO
        self.estimator = self.regressor.fit(data_matrix, values)

        self.mean = numpy.mean(self.estimator.predict(data_matrix))

        self.cal_estimator = self.estimator
        self.status = 'trained'
        return self

    def predict(self, vectorized_graph):
        res = self.cal_estimator.predict(vectorized_graph)
        if self.inverse_prediction:
            res += (self.mean - res) * 2
        return res


class TwoClassEstimator:
    '''
    there might be a bug connected to nx.digraph..
    '''

    def __init__(self, cv=2, n_jobs=-1, recalibrate=True, classifier=SGDClassifier(loss='log')):
        '''
        Parameters
        ----------
        cv:
        n_jobs: jobs for fitting
        recalibrate: recalibration... not implemented, not sure if needed.
        classifier: calssifier object

        Returns
        -------
        '''
        self.status = 'new'
        self.cv = cv
        self.n_jobs = n_jobs
        self.classifier = classifier
        self.inverse_prediction = False
        self.recalibrate = recalibrate

    def fit2(self, data_matrix, data_matrix_neg, random_state=None):
        # TODO MERGE THIS WITH THE FIT METHOD
        if random_state is not None:
            random.seed(random_state)
        # use eden to fitoooOoO

        vecone = map(hashvec,data_matrix)
        vectwo = { a:0 for a in map(hashvec,data_matrix_neg)}
        for e in vecone:
            if e in vectwo:
                print 'same instances in pos/neg set.....(graphlearn/estimator/twoclass)'

        self.estimator = simple_fit_estimator(self.classifier, data_matrix,
                                              data_matrix_neg)

        self.cal_estimator = self.estimator
        self.status = 'trained'
        return self

    def _partial(self, data_matrix, data_matrix_neg, random_state=None,**args):
        '''
        support for partial fitting in GAT

        partial fitting does not work so hot..
        few ideas to try:
        -- warm start instead of partial fit:
            should not work as learning rate decreases but new
            instances should have higher weight than older negatives
        -- partial fit and reprovide the old positives:
            might work...
        '''
        if random_state is not None:
            random.seed(random_state)
        data=vstack((data_matrix,data_matrix_neg))
        data_y=[1]*data_matrix.shape[0]+[-1]*data_matrix_neg.shape[0]
        #self.cal_estimator = SGDClassifier(loss='log', class_weight='balanced') # ballanced will not work :)
        self.cal_estimator = SGDClassifier(loss='log', class_weight={1:9,-1:1},average=True)
        #self.testimator.fit(data_matrix, data_y)
        #self.cal_estimator = CalibratedClassifierCV(self.testimator, cv=self.cv, method='sigmoid')
        #print '*'*80
        #print args
        self.cal_estimator.partial_fit(data, data_y,classes=np.array([1, -1]), **args)
        self.status='trained'
        return self

    def fit(self, data_matrix, data_matrix_neg, random_state=None,**args):
        if random_state is not None:
            random.seed(random_state)
        data=vstack((data_matrix,data_matrix_neg))
        data_y=[1]*data_matrix.shape[0]+[-1]*data_matrix_neg.shape[0]
        #print 'shape:',data.shape
        #self.cal_estimator = SGDClassifier(loss='log', class_weight='balanced') # ballanced will not work :)
        self.cal_estimator = SGDClassifier(loss='log',average=True)
        #self.testimator.fit(data_matrix, data_y)
        #self.cal_estimator = CalibratedClassifierCV(self.testimator, cv=self.cv, method='sigmoid')
        #print '*'*80
        #print args
        self.cal_estimator.fit(data, data_y, **args)
        self.status='trained'
        return self



    def predict(self, vectorized_graph):
        if self.recalibrate:
            result = self.cal_estimator.predict_proba(vectorized_graph)[0, 1]
            #print result
        else:
            print 'if i see this there is a problem, (twoclass estimator)'
            result = self.cal_estimator.decision_function(vectorized_graph)[0]

        if self.inverse_prediction:
            return 1 - result
        return result


class OneClassEstimator:
    '''
    there might be a bug connected to nx.digraph..
    '''

    def __init__(self, nu=.5, cv=2, n_jobs=-1, move_bias_calibrate=True, classifier=SGDClassifier(loss='log')):
        '''
        Parameters
        ----------
        nu: part of graphs that will be placed in the negative set (0~1)
        cv:
        n_jobs: jobs for fitting
        move_bias_calibrate: after moving the bias we can recalibrate
        classifier: calssifier object
        Returns
        -------
        '''
        self.status = 'new'
        self.nu = nu
        self.cv = cv
        self.n_jobs = n_jobs
        self.move_bias_recalibrate = move_bias_calibrate
        self.classifier = classifier
        self.inverse_prediction = False

        self.intercept_ = .5  # PROJECT PRETEND TO BE UNCALLIBRATED TO TRICK EDEN

    # tricking eden th think i am a normal estimator... hehhehe
    #def decision_function(self, vector):  # PROJECT PRETEND TO BE UNCALLIBRATED TO TRICK EDEN
    #    return numpy.array([self.predict_single(sparse) for sparse in vector])
    def decision_function(self, vector):
        # PROJECT PRETEND TO BE UNCALLIBRATED TO TRICK EDEN
        # ok eden annotate is really broken,... needs more trickery on my part..


        # eden will ask (unreasonably) for an intersect array.. .. this hack should work..
        self.intercept_= self.superesti.intercept_

        # so eden expects a 2d array, but will throw away the lesser values.. so we provide what it wants..
        answer =  self.superesti.decision_function(vector)
        return np.vstack((answer, (answer-1))).T

    def fit(self, data_matrix, random_state=None):

        if random_state is not None:
            random.seed(random_state)

        # use eden to fitoooOoO
        self.estimator = self.fit_estimator(data_matrix, n_jobs=self.n_jobs, cv=self.cv, random_state=random_state)

        # move bias to obtain oneclassestimator
        self.cal_estimator = self.move_bias(data_matrix, estimator=self.estimator, nu=self.nu, cv=self.cv)

        self.status = 'trained'
        return self

    '''
    disabled for now.. since the discsampler is not expected to work
    def fit_2(self, pos_iterator, neg_iterator, vectorizer=None, cv=2, n_jobs=-1):
        """
        This is used in the discsampler .,., i am not sure why i am not using eden directly.
        I will fix this when i look into the disk sampler next time.
        :param pos_iterator:
        :param neg_iterator:
        :param vectorizer:
        :param cv:
        :param n_jobs:
        :return:
        """
        self.vectorizer=vectorizer
        data_matrix = vectorizer.fit_transform(pos_iterator)
        neagtive_data_matrix = vectorizer.transform(neg_iterator)
        estimator = eden_fit_estimator(SGDClassifier(loss='log'),
                                       positive_data_matrix=data_matrix,
                                       negative_data_matrix=neagtive_data_matrix,
                                       cv=cv,
                                       n_jobs=n_jobs,
                                       n_iter_search=10)
        # esti= CalibratedClassifierCV(estimator,cv=cv,method='sigmoid')
        # esti.fit( vstack[ X,Y], numpy.asarray([1]*X.shape[0] + [0]*Y.shape[0]))
        return estimator
    '''

    def fit_estimator(self, data_matrix, n_jobs=-1, cv=2, random_state=42):
        '''
        create self.estimator...
        by inversing the data_matrix set to get a negative set
        and then using edens fit_estimator
        '''
        # create negative set:
        data_matrix_neg = data_matrix.multiply(-1)
        # i hope loss is log.. not 100% sure..
        # probably calibration will fix this#
        return simple_fit_estimator(self.classifier, data_matrix,
                                    data_matrix_neg)



    def move_bias(self, data_matrix, estimator=None, nu=.5, cv=2):
        '''
            move bias until nu of data_matrix are in the negative class
            then use scikits calibrate to calibrate self.estimator around the input
        '''
        #  move bias
        # l = [(estimator.decision_function(g)[0], g) for g in data_matrix]
        # l.sort(key=lambda x: x[0])
        # element = int(len(l) * nu)
        # estimator.intercept_ -= l[element][0]

        scores = [estimator.decision_function(sparse_vector)[0]
                  for sparse_vector in data_matrix]
        scores_sorted = sorted(scores)
        pivot = scores_sorted[int(len(scores_sorted) * self.nu)]
        estimator.intercept_ -= pivot

        # calibrate
        if self.move_bias_recalibrate:
            # data_matrix_binary = vstack([a[1] for a in l])
            # data_y = numpy.asarray([0] * element + [1] * (len(l) - element))
            data_y = numpy.asarray([1 if score >= pivot else -1 for score in scores])
            self.testimator = SGDClassifier(loss='log')
            self.testimator.fit(data_matrix, data_y)
            # estimator = CalibratedClassifierCV(estimator, cv=cv, method='sigmoid')
            estimator = CalibratedClassifierCV(self.testimator, cv=cv, method='sigmoid')
            estimator.fit(data_matrix, data_y)
        return estimator

    def predict_single(self, vectorized_graph):
        if self.move_bias_recalibrate:
            result = self.cal_estimator.predict_proba(vectorized_graph)[0, 1]
        else:
            result = self.cal_estimator.decision_function(vectorized_graph)[0]

        if self.inverse_prediction:
            return 1 - result
        return result

    # probably broken ... you should use predict single now o OO
    def predict(self, things):
        return self.predict_single(things)
        #return numpy.array([1 if self.predict_single(thing) > .5 else 0 for thing in things])


class ExperimentalOneClassEstimator:
    '''
    there might be a bug connected to nx.digraph..
    '''

    def __init__(self, nu=.5, cv=2, n_jobs=-1, move_bias_calibrate=True, classifier=SGDClassifier(loss='log')):
        '''
        Parameters
        ----------
        nu: part of graphs that will be placed in the negative set (0~1)
        cv:
        n_jobs: jobs for fitting
        move_bias_calibrate: after moving the bias we can recalibrate
        classifier: calssifier object
        Returns
        -------
        '''
        self.status = 'new'
        self.nu = nu
        self.cv = cv
        self.n_jobs = n_jobs
        self.move_bias_recalibrate = move_bias_calibrate
        self.classifier = classifier
        self.inverse_prediction = False

        #self.intercept_ = .5  # PROJECT PRETEND TO BE UNCALLIBRATED TO TRICK EDEN

    # tricking eden th think i am a normal estimator... hehhehe
    def decision_function(self, vector):
        # PROJECT PRETEND TO BE UNCALLIBRATED TO TRICK EDEN
        # ok eden annotate is really broken,... needs more trickery on my part..


        # eden will ask (unreasonably) for an intersect array.. .. this hack should work..
        self.intercept_= self.superesti.intercept_

        # so eden expects a 2d array, but will throw away the lesser values.. so we provide what it wants..
        answer =  self.superesti.decision_function(vector)
        return np.vstack((answer, (answer-1))).T



    def fit(self, data_matrix, random_state=None):

        if random_state is not None:
            random.seed(random_state)

        # use eden to fitoooOoO
        self.estimator = self.fit_estimator(data_matrix, n_jobs=self.n_jobs, cv=self.cv, random_state=random_state)

        # move bias to obtain oneclassestimator
        self.cal_estimator = self.move_bias(data_matrix, estimator=self.estimator, nu=self.nu, cv=self.cv)

        self.status = 'trained'
        return self

    '''
    disabled for now.. since the discsampler is not expected to work
    def fit_2(self, pos_iterator, neg_iterator, vectorizer=None, cv=2, n_jobs=-1):
        """
        This is used in the discsampler .,., i am not sure why i am not using eden directly.
        I will fix this when i look into the disk sampler next time.
        :param pos_iterator:
        :param neg_iterator:
        :param vectorizer:
        :param cv:
        :param n_jobs:
        :return:
        """
        self.vectorizer=vectorizer
        data_matrix = vectorizer.fit_transform(pos_iterator)
        neagtive_data_matrix = vectorizer.transform(neg_iterator)
        estimator = eden_fit_estimator(SGDClassifier(loss='log'),
                                       positive_data_matrix=data_matrix,
                                       negative_data_matrix=neagtive_data_matrix,
                                       cv=cv,
                                       n_jobs=n_jobs,
                                       n_iter_search=10)
        # esti= CalibratedClassifierCV(estimator,cv=cv,method='sigmoid')
        # esti.fit( vstack[ X,Y], numpy.asarray([1]*X.shape[0] + [0]*Y.shape[0]))
        return estimator
    '''

    def fit_estimator(self, data_matrix, n_jobs=-1, cv=2, random_state=42):
        '''
        create self.estimator...
        by inversing the data_matrix set to get a negative set
        and then using edens fit_estimator
        '''
        # create negative set:
        data_matrix_neg = data_matrix.multiply(-1)
        # i hope loss is log.. not 100% sure..
        # probably calibration will fix this#
        return simple_fit_estimator(self.classifier, data_matrix, data_matrix_neg)

    def move_bias(self, data_matrix, estimator=None, nu=.5, cv=2):
        '''
            move bias until nu of data_matrix are in the negative class
            then use scikits calibrate to calibrate self.estimator around the input
        '''
        #  move bias
        # l = [(estimator.decision_function(g)[0], g) for g in data_matrix]
        # l.sort(key=lambda x: x[0])
        # element = int(len(l) * nu)
        # estimator.intercept_ -= l[element][0]

        scores = [estimator.decision_function(sparse_vector)[0]
                  for sparse_vector in data_matrix]
        scores_sorted = sorted(scores)
        pivot = scores_sorted[int(len(scores_sorted) * self.nu)]
        estimator.intercept_ -= pivot

        # calibrate
        if self.move_bias_recalibrate:
            # data_matrix_binary = vstack([a[1] for a in l])
            # data_y = numpy.asarray([0] * element + [1] * (len(l) - element))
            data_y = numpy.asarray([1 if score >= pivot else -1 for score in scores])
            self.superesti = SGDClassifier(loss='log')  #
            self.superesti.fit(data_matrix, data_y)
            # estimator = CalibratedClassifierCV(estimator, cv=cv, method='sigmoid')
            # estimator = CalibratedClassifierCV(self.testimator, cv=cv, method='sigmoid')
            # estimator.fit(data_matrix, data_y)

            self.intercept_= self.superesti.intercept_

        return self.superesti

    def predict_single(self, vectorized_graph):

        return self.superesti.decision_function(vectorized_graph)[0]

    # probably broken ... you should use predict single now o OO
    def predict(self, things):
        # return self.predict_single(things)
        # return numpy.array( [ 1 if self.predict_single(thing)>.5 else 0 for thing in things] )
        return self.superesti.predict(things)
