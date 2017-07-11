#!/usr/bin/python

text='''verbose=1 # sets verbose level, is not passed to sampler
start_graphs="asd",
num_graphs=10,# number of graphs used
model="model.gs",
out="sdf",
graph_iter=None,
probabilistic_core_choice=True,
score_core_choice=False,
max_size_diff=-1,
similarity=-1,
n_samples=None,
proposal_probability=False,
batch_size=10,
n_jobs=0,
target_orig_cip=False,
n_steps=50,
quick_skip_orig_cip=False,
improving_threshold=-1,
improving_linear_start=0,
accept_static_penalty=0.0,
accept_min_similarity=0.0,
select_cip_max_tries=20,
burnin=0,
backtrack=0,
include_seed=False,
keep_duplicates=False,
monitor=False,
init_only=False,'''


# arg parse busines
import makeparser
parser= makeparser.makeparser(text)

if __name__ == "__main__":

    args=vars(parser.parse_args())
    import os.path
    if not os.path.isfile(args['start_graphs']):
        parser.print_usage()
        print 'at least provide a path to input'
        exit()

    print "*raw args"
    print "*" * 80
    print args

    # verbosity
    from eden.util import configure_logging
    import logging
    configure_logging(logging.getLogger(),verbosity=args['verbose'])
    args.pop('verbose')

    # graphs
    from eden.io.gspan import gspan_to_eden
    from itertools import islice
    args['graph_iter'] = islice(gspan_to_eden(args.pop('start_graphs')),args.pop('num_graphs'))


    #output
    OUTFILE=args.pop('out')
    MODEL=args.pop('model')

    # CREATE SAMPLER
    from graphlearn01.graphlearn import Sampler
    s=Sampler()
    s.load(MODEL)
    results=s.transform(**args)


    import graphlearn01.utils.draw_openbabel as ob

    for i,samplepath in enumerate(results):
        for j,graph in enumerate(samplepath):
            with open(str(i)+'.'+str(j)+'.'+OUTFILE,'w') as f:
                f.write(ob.graph_to_molfile(graph))

