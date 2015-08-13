from eden.util import fit_estimator as eden_fit_estimator
import numpy
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
import random


class EstimatorWrapper:

    '''
    graphlearn will edata_matrixpect fit to return an estimator that is used in the graphlearn
    (if you use sampler.fit)
    '''

    def fit(self, graphs, vectorizer=None, nu=.5, cv=2, n_jobs=-1, random_state=None):
        if random_state is not None:
            random.seed(random_state)
        data_matrix = vectorizer.fit_transform(graphs)
        self.estimator = self.fit_estimator(data_matrix, n_jobs=n_jobs, cv=cv, random_state=random_state)
        cal_estimator = self.calibrate_estimator(data_matrix, estimator=self.estimator, nu=nu, cv=cv)
        return cal_estimator

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
        return eden_fit_estimator(SGDClassifier(loss='log'), positive_data_matrix=data_matrix,
                                  negative_data_matrix=data_matrix_neg,
                                  cv=cv,
                                  n_jobs=n_jobs,
                                  n_iter_search=10,
                                  random_state=random_state)

    def calibrate_estimator(self, data_matrix, estimator=None, nu=.5, cv=2):
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
        data_matrix_binary = vstack([a[1] for a in l])
        data_y = numpy.asarray([0] * element + [1] * (len(l) - element))
        estimator = CalibratedClassifierCV(estimator, cv=cv, method='sigmoid')
        estimator.fit(data_matrix_binary, data_y)

        return estimator
