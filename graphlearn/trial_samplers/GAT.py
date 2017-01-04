from graphlearn.graphlearn import Sampler
from graphlearn.estimate import TwoClassEstimator as TwoClass
import copy



from scipy.sparse import vstack

def get_sampler():
    return Sampler(
        # vectorizer=eden.graph.Vectorizer(complexity=3,n_jobs=2),
        random_state=None,
        # estimator=estimate.OneClassEstimator(nu=.5, cv=2, n_jobs=-1),
        # graphtransformer=transform.GraphTransformer(),
        # feasibility_checker=feasibility.FeasibilityChecker(),
        # decomposer=decompose.Decomposer(node_entity_check=lambda x, y:True, nbit=20),
        # grammar=LocalSubstitutableGraphGrammar(radius_list=[0,1], thickness_list=[1,2], min_cip_count=2,min_interface_count=2),
        n_steps=30,
        n_samples=2,
        core_choice_byfrequency=False,
        core_choice_byscore=False,
        core_choice_bytrial=True,
        core_choice_bytrial_multiplier=1.3,
        size_diff_core_filter=4,  # BROKEN ?
        burnin=10,
        include_seed=False,
        proposal_probability=False,
        improving_threshold_fraction=.5,
        improving_linear_start_fraction=0.0,
        accept_static_penalty=0.0,
        n_jobs=1,
        select_cip_max_tries=100,
        keep_duplicates=False,
        monitor=False)


def flatten(thing):
    return [i for e in thing for i in e]

def get_sample_weights(pos, genlist):
    res = []
    cweight = 1.0
    for sublist in reversed(genlist):
        if sublist:
            res.append([cweight] * len(sublist))
            cweight /= 2

    res = flatten(reversed(res))
    res = [1] * len(pos) + res
    return res



def graphs_to_vectors(sampler,graphs):
    decomp=sampler.fit_make_decomposers(graphs)
    return sampler.decomps_to_vectors(decomp)

def sample(sampler, n_iterations= 3, seedgraphs= None ):


    # fit initial 1 class svm
    seed_decomposers = sampler.fit_make_decomposers(seedgraphs)
    seed_vectors = sampler.decomps_to_vectors(seed_decomposers)
    sampler.fit_grammar(seed_decomposers)
    sampler.fit_estimator(seed_decomposers)


    # save constructed graphs...
    constructed_graphs = [[]]
    constructed_vectors = [[]]
    constructed_graphs.append(flatten(sampler.transform(seedgraphs)))
    constructed_vectors.append(graphs_to_vectors(sampler,constructed_graphs[-1]))

    # save estimator
    estimators=[]
    estimators.append(copy.deepcopy(sampler.estimatorobject))

    # prepare 2 class svm
    sampler.estimatorobject = TwoClass(cv=5,recalibrate=True)
    new_decomposers = sampler.fit_make_decomposers(constructed_graphs[-1])
    sampler.fit_estimator(seed_decomposers, new_decomposers)

    def make_partial_fit_negs(graphs):
        vectorz= sampler.vectorizer.transform(map( lambda x: x.pre_vectorizer_graph(),sampler.fit_make_decomposers(graphs)))
        return  vectorz,[-1]*len(graphs)
    print 'loopstart'
    for i in range(n_iterations):
        # construct
        constructed_graphs.append(flatten(sampler.transform(seedgraphs)))
        constructed_vectors.append(graphs_to_vectors(sampler,constructed_graphs[-1]))

        # partial fit:
        #a,b=make_partial_fit_negs(constructed_graphs[-1])
        #sampler.estimatorobject.cal_estimator.partial_fit(a,b, sample_weight=[1]*len(constructed_graphs[-1]))
        # new fit

        weights=get_sample_weights(seedgraphs,constructed_graphs)
        #print len(weights),seed_vectors.shape, vstack(constructed_vectors).shape, len(flatten(constructed_graphs))
        sampler.estimatorobject.fit(seed_vectors,vstack(constructed_vectors[1:]),sample_weight=weights)


        # save esti
        estimators.append(copy.deepcopy(sampler.estimatorobject.cal_estimator))

    return estimators, constructed_vectors, seed_vectors