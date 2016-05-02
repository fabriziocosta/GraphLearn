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


########################################################################
# BUILD ARGPARSER BASED ON THAT OPTIONLIST
########################################################################

# so the optionlist, has equal signs and sometimes '#' letters.
# -> split according to those
tmp=[]
for line in text.split("\n"):
    hel='no help'
    hindex=line.find("#")
    if hindex != -1:
        hel=line[hindex:].strip()
        line=line[:hindex]
    deli = line.find('=')
    if deli ==- 1:
        tmp.append((line[:-1],'',hel))
    else:
        tmp.append((line[:deli],line[deli+1:-1].strip(),hel))
used_names=[]


# optionlist gives a long name for the parameters
# short names are nicer for users
# we try to guess a short name here.
def shorten(name):
    #1. try:split by underscore, use first letters
    #2. try:use longname[:3]
    #3. use longname
    l=name.split("_")
    shortname=''.join([e[0] for e in l])
    if shortname not in used_names and len(shortname)>1:
        used_names.append(shortname)
        return shortname
    shortname = name[:3]
    if shortname not in used_names:
        used_names.append(shortname)
        return shortname
    return name



        
# making a parser... 
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
for arg, value ,helpmsg in tmp:
    # so,, what we need is, long name, short name, type,help(lol), default

    longname = arg
    shortname=shorten(longname)
    value=eval(value)
    typ=type(value)
    default=value 

    # handling list: lists are handled weirdly in argparse,
    # so here is another exception :) 
    nargs = "+" if typ==list else None
    if typ==list:
        typ=int
        
    parser.add_argument(
            "--"+shortname,
            "--"+longname,
	        nargs=nargs,
            dest=longname,
            type=typ,
            help=helpmsg,
            default=default) 




########################################################################
# -> READ OPTIONS, PREPARE INSTANCES AND FIT A SAMPLER
########################################################################
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
from graphlearn.graphlearn import Sampler
s=Sampler(**args)
print "*fit"
print "*"*80
print fitargs
s.fit(**fitargs)
s.save(OUTFILE)
