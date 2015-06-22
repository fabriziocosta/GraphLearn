from eden.util import fit_estimator
import numpy
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier


class estimator:
    '''
    graphlearn will expect fit to return an estimator that is used in the graphlearn.. (if you use sampler.fit)
    '''

    def fit(self, graphs, vectorizer=None, nu=.5, cv=2, n_jobs=-1):
        X = vectorizer.transform(graphs)
        estimator = self.fit_estimator(X, n_jobs=n_jobs, cv=cv)
        cal_estimator = self.calibrate_estimator(X, estimator=estimator, nu=nu, cv=cv)
        return cal_estimator

    def fit_2(self, ipos, ineg, vectorizer=None, cv=2, n_jobs=-1):
        X = vectorizer.transform(ipos)
        Y = vectorizer.transform(ineg)
        estimator = fit_estimator(SGDClassifier(loss='log'), positive_data_matrix=X, negative_data_matrix=Y, cv=cv,
                                  n_jobs=n_jobs, n_iter_search=10)
        # esti= CalibratedClassifierCV(estimator,cv=cv,method='sigmoid')
        # esti.fit( vstack[ X,Y], numpy.asarray([1]*X.shape[0] + [0]*Y.shape[0]))
        return estimator

    def fit_estimator(self, X, n_jobs=-1, cv=2):
        '''
        create self.estimator...
        by inversing the X set to get a negative set
        and then using edens fit_estimator
        '''

        # create negative set:
        X_neg = X.multiply(-1)
        # i hope loss is log.. not 100% sure..
        # probably calibration will fix this#
        return fit_estimator(SGDClassifier(loss='log'), positive_data_matrix=X,
                             negative_data_matrix=X_neg,
                             cv=cv,
                             n_jobs=n_jobs,
                             n_iter_search=10)

    def calibrate_estimator(self, X, estimator=None, nu=.5, cv=2):
        '''
            move bias until nu of X are in the negative class

            then use scikits calibrate to calibrate self.estimator around the input
        '''
        #  move bias
        l = [(estimator.decision_function(g)[0], g) for g in X]
        l.sort(key=lambda x: x[0])
        element = int(len(l) * nu)
        estimator.intercept_ -= l[element][0]

        # calibrate
        data_matrix = vstack([a[1] for a in l])
        data_y = numpy.asarray([0] * element + [1] * (len(l) - element))
        estimator = CalibratedClassifierCV(estimator, cv=cv, method='sigmoid')
        estimator.fit(data_matrix, data_y)

        return estimator
