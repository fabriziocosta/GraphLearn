from eden.util import fit_estimator as eden_fit_estimator
import numpy
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier




'''
graphlearn will expect fit to return an estimator that is used in the graphlearn.. (if you use sampler.fit)
'''
def fit(Graph_iter,vectorizer,nu=.5,n_jobs=-1):
    X = vectorizer.transform(Graph_iter)
    esti = fit_estimator(X, n_jobs)
    cesti = calibrate_estimator(X,esti,nu)
    return cesti





def fit_estimator( X, n_jobs=-1, cv=10):
    '''
    create self.estimator...
    by inversing the X set to get a negative set
    and then using edens fit_estimator
    '''

    # get negative set:
    X_neg = X.multiply(-1)
    # i hope loss is log.. not 100% sure..
    # probably calibration will fix this#
    return eden_fit_estimator(SGDClassifier(), positive_data_matrix=X,
                                        negative_data_matrix=X_neg,
                                        cv=cv,
                                        n_jobs=n_jobs,
                                        verbose=0,
                                        n_iter_search=10)

def calibrate_estimator( X,estimator, nu=.5):
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
    estimator = CalibratedClassifierCV(estimator, cv=3, method='sigmoid')
    estimator.fit(data_matrix, data_y)

    return estimator
