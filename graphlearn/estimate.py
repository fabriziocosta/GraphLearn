'''
OneClassEstimator
'''
from eden.util import fit_estimator as eden_fit_estimator
import numpy
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
import random
import networkx as nx
from utils import draw


class TwoClassEstimator:
    '''
    there might be a bug connected to nx.digraph..
    '''
    def __init__(self, cv=2, n_jobs=-1,recalibrate=False, classifier=SGDClassifier(loss='log')):
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
        self.classifier=classifier
        self.inverse_prediction = False
        self.recalibrate=recalibrate

    def fit(self, data_matrix,data_matrix_neg, random_state=None):
        if random_state is not None:
            random.seed(random_state)
        # use eden to fitoooOoO
        self.estimator = eden_fit_estimator(self.classifier, positive_data_matrix=data_matrix,
                           negative_data_matrix=data_matrix_neg,
                           cv=self.cv,
                           n_jobs=self.n_jobs,
                           n_iter_search=10,
                           random_state=random_state)
        self.cal_estimator=self.estimator
        self.status = 'trained'
        return self

    def predict(self, vectorized_graph):
        if self.recalibrate:
            result = self.cal_estimator.predict_proba(vectorized_graph)[0, 1]
        else:
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
            return eden_fit_estimator(self.classifier, positive_data_matrix=data_matrix,
                                      negative_data_matrix=data_matrix_neg,
                                      cv=cv,
                                      n_jobs=n_jobs,
                                      n_iter_search=10,
                                      random_state=random_state)

        def move_bias(self, data_matrix, estimator=None, nu=.5, cv=2):
            '''
                move bias until nu of data_matrix are in the negative class
                then use scikits calibrate to calibrate self.estimator around the input
            '''
            #  move bias
            l = [(estimator.decision_function(g)[0], g) for g in data_matrix]
            l.sort(key=lambda x: x[0])
            element = int(len(l) * nu)
            estimator.intercept_ -= l[element][0]

            # calibrate
            if self.move_bias_recalibrate:
                data_matrix_binary = vstack([a[1] for a in l])
                data_y = numpy.asarray([0] * element + [1] * (len(l) - element))
                estimator = CalibratedClassifierCV(estimator, cv=cv, method='sigmoid')
                estimator.fit(data_matrix_binary, data_y)
            return estimator

        def predict(self, vectorized_graph):
            if self.move_bias_recalibrate:
                result = self.cal_estimator.predict_proba(vectorized_graph)[0, 1]
            else:
                result = self.cal_estimator.decision_function(vectorized_graph)[0]

            if self.inverse_prediction:
                return 1 - result
            return result