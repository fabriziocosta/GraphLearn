# do argparse things:
import graphlearn_sample_argparser as generated
argp=generated.parser
args=vars(argp.parse_args())
print args

# verbosity
from eden.util import configure_logging
import logging
configure_logging(logging.getLogger(),verbosity=args['verbose'])
args.pop('verbose')

#output
OUTFILE=args.pop('out')
MODEL=args.pop('model')

# CREATE SAMPLER
from graphlearn.graphlearn import Sampler
s=Sampler()
s.load(MODEL)
s.sample(***args)
s.save(OUTFILE)

