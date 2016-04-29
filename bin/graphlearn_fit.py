# do argparse things:
import graphlearn_fit_argparser as generated
argp=generated.parser
args=vars(argp.parse_args())
print args

# verbosity
from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=args['verbose'])
args.pop('verbose')

# handle Vectorizer:
from eden.graph import Vectorizer
args['vectorizer'] = Vectorizer(args.pop('vectorizer_complexity'))

# for these we just use defaults...
#graphtransformer=transform.GraphTransformer(),
#feasibility_checker=feasibility.FeasibilityChecker(),
#decomposergen=decompose.Decomposer,

# estimator, if the user is providing a negative graph set, we use
# the twoclass esti OO
import graphlearn.estimate as estimate
if args['negative_input']==None:
    args['estimator']=estimate.OneClassEstimator(nu=.5, cv=2, n_jobs=-1)
else:
    args['estimator']=estimate.TwoClassEstimator( cv=2, n_jobs=-1)
    

#args for fitting:
from eden.converter.graph.gspan import gspan_to_eden
from itertools import islice
fitargs={ k:args.pop(k) for k in ['lsgg_include_negatives','grammar_n_jobs','grammar_batch_size']}
if args['negative_input']!=None:
    fitargs['negative_input'] = islice(gspan_to_eden(args.pop('negative_input')),args['num_graphs_neg'])
else:
    args.pop('negative_input')
    args.pop('num_graphs_neg')
fitargs['input'] = islice(gspan_to_eden(args.pop('input')),args.pop('num_graphs'))

#output
OUTFILE=args.pop('output')

print "*"*80
print args

# CREATE SAMPLER
from graphlearn.graphlearn import Sampler
s=Sampler(**args)
s.fit(**fitargs)
s.save(OUTFILE)

