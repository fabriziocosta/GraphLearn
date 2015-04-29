import joblib
import utils.myeden as graphlearn_utils
from eden.util import fit_estimator as eden_fit_estimator
import networkx as nx
import itertools
import random
from multiprocessing import Pool, Manager
from eden.graph import Vectorizer
from networkx.algorithms import isomorphism as iso
from eden import fast_hash
import utils.draw as draw
import logging
import numpy
import dill
import eden
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier



# ############################### FEASIBILITY CHECKER ###################


class FeasibilityChecker():

    def __init__(self,logger):
        self.logger=logger
        self.checklist = []
        self.checklist.append(defaultcheck)

    def check(self, ng):
        for f in self.checklist:
            if f(ng) == False:

                return False
        return True


def defaultcheck(ng):
    if len(ng) < 1:
        self.logger.debug('graph non existent')
        return False
    for node_id in ng.nodes_iter():
        if 'edge' in ng.node[node_id]:
            if len(ng.neighbors(node_id)) != 2:
                self.logger.debug('feasibility- edge check failed, (interface twist phenomenon probably)')
                return False
    return True
