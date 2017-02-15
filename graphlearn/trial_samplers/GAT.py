from graphlearn.graphlearn import Sampler
from graphlearn.estimate import TwoClassEstimator as TwoClass
import copy
import graphlearn.minor.molecule.transform_cycle as mol
import graphlearn.minor.decompose as decompose
from graphlearn import feasibility
from scipy.sparse import vstack

from eden.graph import Vectorizer

def get_sampler():
    return Sampler(
        #USE THIS:
        #graphtransformer=mol.GraphTransformerCircles(),
        #decomposer = decompose.MinorDecomposer(),

        # OR THIS:
        feasibility_checker=feasibility.cycle_feasibility_checker(6),

        #  some default stuff
        #grammar=LocalSubstitutableGraphGrammar(radius_list=[0,1], thickness_list=[1,2], min_cip_count=2,min_interface_count=2),
        # vectorizer=eden.graph.Vectorizer(complexity=3,n_jobs=2),
        # estimator=estimate.OneClassEstimator(nu=.5, cv=2, n_jobs=-1),
        # feasibility_checker=feasibility.FeasibilityChecker(),
        #graphtransformer=transform.GraphTransformer(),
        #decomposer=decompose.Decomposer(node_entity_check=lambda x, y:True, nbit=20),
        vectorizer=Vectorizer(r=3,d=3),
        random_state=None,
        n_steps=30,
        n_samples=2,
        core_choice_byfrequency=True,
        core_choice_byscore=False,
        core_choice_bytrial=False,
        core_choice_bytrial_multiplier=1.3,
        size_diff_core_filter=4,
        burnin=10,
        include_seed=False,
        proposal_probability=False,
        improving_threshold_fraction=.5,
        improving_linear_start_fraction=0.0,
        accept_static_penalty=0.0,
        n_jobs=8,
        select_cip_max_tries=100,
        keep_duplicates=False,
        monitor=False)


def flatten(thing):
    return [i for e in thing for i in e]


def get_sample_weights2(pos,genlist):

    res=[]
    for graphs in genlist:
        res+=[-1]*len(graphs)

    res+= [ float(len(res))/len(pos)  ]*len(pos)
    print res
    return res

def get_sample_weights(pos, genlist):
    res = []
    cweight = 1.0

    weight_of_pos_instances=0.0

    for sublist in reversed(genlist):
        if sublist:
            res.append([cweight] * len(sublist))
            # this is a little bit like 1+.5+.25+.125 etc
            weight_of_pos_instances += (float(len(sublist))/len(pos))*cweight
            cweight /= 2

    res = flatten(reversed(res))
    res = [weight_of_pos_instances] * len(pos) + res
    return res



def graphs_to_vectors(sampler,graphs):
    decomp=sampler.fit_make_decomposers(graphs)
    return sampler.decomps_to_vectors(decomp)

def generative_adersarial_training(sampler, n_iterations= 3, seedgraphs= None, partial_estimator=True):


    # fit initial 1 class svm
    seed_decomposers = sampler.fit_make_decomposers(seedgraphs)
    seed_vectors = sampler.decomps_to_vectors(seed_decomposers)
    sampler.fit_grammar(seed_decomposers)
    sampler.fit_estimator(seed_decomposers)


    # save constructed graphs...
    constructed_graphs = []
    constructed_vectors = []
    constructed_graphs.append(flatten(sampler.transform(seedgraphs)))
    constructed_vectors.append(graphs_to_vectors(sampler,constructed_graphs[-1]))

    # save estimator
    estimators=[]
    estimators.append(copy.deepcopy(sampler.estimatorobject.cal_estimator))

    # prepare 2 class svm
    sampler.estimatorobject = TwoClass(cv=5,recalibrate=True)



    new_decomposers = sampler.fit_make_decomposers(constructed_graphs[-1])
    if partial_estimator:
        sampler.estimatorobject._partial(sampler.decomps_to_vectors(seed_decomposers),sampler.decomps_to_vectors(new_decomposers))
    else:
        sampler.fit_estimator(seed_decomposers, new_decomposers)

    def make_partial_fit_args(graphs):
        # return X, y
        # with old and new
        vectors= sampler.vectorizer.transform(map( lambda x: x.pre_vectorizer_graph(),sampler.fit_make_decomposers(graphs)))
        X= vstack((vectors, seed_vectors))
        return  X, [-1]*len(graphs)+[1]*len(seedgraphs)

    for i in range(n_iterations):
        # construct
        constructed_graphs.append(flatten(sampler.transform(seedgraphs)))
        if len(constructed_graphs[-1]) == 0:
            print "NO GRAPHZ GENERATED,, eden will die soon"
        constructed_vectors.append(graphs_to_vectors(sampler,constructed_graphs[-1]))

        # partial fit:
        #a,b=make_partial_fit_negs(constructed_graphs[-1])
        #sampler.estimatorobject.cal_estimator.partial_fit(a,b, sample_weight=[1]*len(constructed_graphs[-1]))
        # new fit
        if partial_estimator:
            X,y =make_partial_fit_args(constructed_graphs[-1])
            sampler.estimatorobject.cal_estimator.partial_fit(X,y)
        else:
            weights=get_sample_weights(seedgraphs,constructed_graphs)
            #print len(weights),seed_vectors.shape, vstack(constructed_vectors).shape, len(flatten(constructed_graphs))
            #sampler.estimatorobject.fit(seed_vectors,vstack(constructed_vectors[1:]),sample_weight=weights)
            sampler.estimatorobject.fit(seed_vectors,vstack(constructed_vectors),sample_weight=weights)

        # save esti
        estimators.append(copy.deepcopy(sampler.estimatorobject.cal_estimator))
        print '%d ' % i,

    return estimators, constructed_graphs #, seed_vectors












#######################################################
########################################################
# this is the hack version


def generative_adersarial_training_HACK(sampler, n_iterations= 3, seedgraphs= None,neg_vectors=None, partial_estimator=True):

    # fit initial 1 class svm
    seed_decomposers = sampler.fit_make_decomposers(seedgraphs)
    seed_vectors = sampler.decomps_to_vectors(seed_decomposers)

    #neg_decomposers = sampler.fit_make_decomposers(negatives)
    #neg_vectors = sampler.decomps_to_vectors(neg_decomposers)


    sampler.fit_grammar(seed_decomposers)
    #sampler.fit_estimator(seed_decomposers)


    # prepare 2 class svm
    sampler.estimatorobject = TwoClass(cv=5,recalibrate=True)
    sampler.estimatorobject.fit(seed_vectors,neg_vectors)


    # save constructed graphs...
    constructed_graphs = []
    constructed_vectors = []
    constructed_graphs.append(flatten(sampler.transform(seedgraphs)))
    constructed_vectors.append(graphs_to_vectors(sampler,constructed_graphs[-1]))


    # save estimator
    estimators=[]
    estimators.append(copy.deepcopy(sampler.estimatorobject.cal_estimator))


    new_decomposers = sampler.fit_make_decomposers(constructed_graphs[-1])
    if partial_estimator:
        sampler.estimatorobject._partial(sampler.decomps_to_vectors(seed_decomposers),sampler.decomps_to_vectors(new_decomposers))
    else:
        sampler.fit_estimator(seed_decomposers, new_decomposers)

    def make_partial_fit_args(graphs):
        # return X, y
        # with old and new
        vectors= sampler.vectorizer.transform(map( lambda x: x.pre_vectorizer_graph(),sampler.fit_make_decomposers(graphs)))
        X= vstack((vectors, seed_vectors))
        return  X, [-1]*len(graphs)+[1]*len(seedgraphs)

    for i in range(n_iterations):
        # construct
        constructed_graphs.append(flatten(sampler.transform(seedgraphs)))
        if len(constructed_graphs[-1]) == 0:
            print "NO GRAPHZ GENERATED,, eden will die soon"
        constructed_vectors.append(graphs_to_vectors(sampler,constructed_graphs[-1]))

        # partial fit:
        #a,b=make_partial_fit_negs(constructed_graphs[-1])
        #sampler.estimatorobject.cal_estimator.partial_fit(a,b, sample_weight=[1]*len(constructed_graphs[-1]))
        # new fit
        if partial_estimator:
            X,y =make_partial_fit_args(constructed_graphs[-1])
            sampler.estimatorobject.cal_estimator.partial_fit(X,y)
        else:
            weights=get_sample_weights(seedgraphs,constructed_graphs)
            #print len(weights),seed_vectors.shape, vstack(constructed_vectors).shape, len(flatten(constructed_graphs))
            #sampler.estimatorobject.fit(seed_vectors,vstack(constructed_vectors[1:]),sample_weight=weights)
            sampler.estimatorobject.fit(seed_vectors,vstack(constructed_vectors),sample_weight=weights)

        # save esti
        estimators.append(copy.deepcopy(sampler.estimatorobject.cal_estimator))
        print '%d ' % i,

    return estimators, constructed_graphs #, seed_vectors


################################################################
################################################################
# THIS IS THE PROPPER VERSION THAT WILL BE CREATED LATER>>>
def generative_adersarial_training_realneg(sampler, n_iterations= 3, seedgraphs= None,negatives=None, partial_estimator=True):


    # fit grammar, make vectors
    seed_vectors = sampler.decomps_to_vectors(seed_decomposers)
    seed_decomposers = sampler.fit_make_decomposers(seedgraphs)
    sampler.fit_grammar(seed_decomposers)

    # save results
    constructed_graphs = []
    constructed_vectors = []
    estimators=[]

    sampler.estimatorobject = TwoClass(cv=5,recalibrate=True)




    for i in range(n_iterations):
        #new_decomposers = sampler.fit_make_decomposers(constructed_graphs[-1])



        # fitting the estimator object
        # sampler.fit_estimator(seed_decomposers, new_decomposers)

        weights=get_sample_weights(seedgraphs,constructed_graphs)
        sampler.estimatorobject.fit(seed_vectors,vstack(constructed_vectors),sample_weight=weights)


        # make graphs
        constructed_graphs.append(flatten(sampler.transform(seedgraphs)))
        if len(constructed_graphs[-1]) == 0:
            print "NO GRAPHZ GENERATED,, eden will die soon"
        constructed_vectors.append(graphs_to_vectors(sampler,constructed_graphs[-1]))




        # save esti
        estimators.append(copy.deepcopy(sampler.estimatorobject.cal_estimator))
        print '%d ' % i,

    return estimators, constructed_graphs #, seed_vectors


