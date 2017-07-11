#!/usr/bin/python
text='''verbose=1 # sets verbose level, is not passed to sampler
nbit=20,
random_state=None,
vectorizer_complexity=3,
radius_list=[0, 1],#asd
thickness_list=[1, 2],
min_cip_count=2,
min_interface_count=2,
input="asd.graph",
output="model.gs",
negative_input=None,
lsgg_include_negatives=False,
grammar_n_jobs=-1,
grammar_batch_size=10,
num_graphs=200 #limit number of graphs read by data sources
num_graphs_neg=200'''


# I excluded these so defaults will be used :) 
#graphtransformer=transform.GraphTransformer(),
#feasibility_checker=feasibility.FeasibilityChecker(),
#decomposergen=decompose.Decomposer,
# grammar=None,


import makeparser
parser=makeparser.makeparser(text)


if __name__ == "__main__":
        
    # do argparse things:
    args=vars(parser.parse_args())

    import os.path
    if not os.path.isfile(args['input']):
        parser.print_usage()
        print 'at least provide a path to input'
        exit()

    print "*raw args"
    print "*"*80
    print args


    # verbosity
    from eden.util import configure_logging
    import logging
    configure_logging(logging.getLogger(),verbosity=args.pop('verbose'))


    # handle Vectorizer:
    from eden.graph import Vectorizer
    args['vectorizer'] = Vectorizer(args.pop('vectorizer_complexity'))


    # estimator, if the user is providing a negative graph set, we use
    # the twoclass esti OO
    import graphlearn01.estimate as estimate
    if args['negative_input']==None:
        args['estimator']=estimate.OneClassEstimator(nu=.5, cv=2, n_jobs=-1)
    else:
        args['estimator']=estimate.TwoClassEstimator( cv=2, n_jobs=-1)
        
    #args for fitting:
    from eden.io.gspan import gspan_to_eden
    from itertools import islice
    fitargs={ k:args.pop(k) for k in ['lsgg_include_negatives','grammar_n_jobs','grammar_batch_size']}

    if args['negative_input']!=None:
        fitargs['negative_input'] = islice(gspan_to_eden(args.pop('negative_input')),args.pop('num_graphs_neg'))
    else:
        args.pop('negative_input')
        args.pop('num_graphs_neg')

    fitargs['input'] = islice(gspan_to_eden(args.pop('input')),args.pop('num_graphs'))

    #output
    OUTFILE=args.pop('output')

    print "*Sampler init"
    print "*"*80
    print args

    # CREATE SAMPLER, dumping the rest of the parsed args :) 
    from graphlearn01.graphlearn import Sampler
    s=Sampler(**args)
    print "*fit"
    print "*"*80
    print fitargs
    s.fit(**fitargs)
    s.save(OUTFILE)
