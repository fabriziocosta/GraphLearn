# get data
from itertools import islice
'''
Create a function that is able to deliver a graph iterator
'''
from eden.converter.fasta import fasta_to_sequence
from eden.converter.rna.rnafold import rnafold_to_eden

from eden.graph import Vectorizer
def rfam_uri(family_id):
    return 'http://rfam.xfam.org/family/%s/alignment?acc=%s&format=fastau&download=0'%(family_id,family_id)


def clean(graphs):
    for g in graphs:
        for n,d in g.nodes(data=True):
            if 'weight' in d:
                d.pop('weight')
        yield g


def rfam_uri(family_id):
    return '%s.fa'%(family_id)

def get_graphs(rfam_id = '../example/RF00005',size=9999):
    seqs = fasta_to_sequence(rfam_uri(rfam_id))
    graphs = islice( clean(rnafold_to_eden(seqs, shape_type=5, energy_range=30, max_num=3)), size)
    return graphs


from eden.converter.fasta import fasta_to_sequence
import itertools


def get_sequences(size=9999):
    sequences = itertools.islice( fasta_to_sequence("../example/RF00005.fa"), size)
    return [ b for (a,b) in sequences ]

def get_sequences_with_names(size=9999):
    sequences = itertools.islice( fasta_to_sequence("../example/RF00005.fa"), size)
    return sequences





'''
learning a grammar
'''
import graphlearn.abstract_graphs.learned_RNA as learned
import graphlearn.abstract_graphs.RNA as rna
from graphlearn import feasibility
feas=feasibility.FeasibilityChecker(checklist=[feasibility.default_check,rna.is_rna])
graphs = get_sequences_with_names(150)
pp=learned.RnaPreProcessor(base_thickness_list=[2],kmeans_clusters=3,structure_mod=False)
sampler=rna.AbstractSampler(radius_list=[0,1],thickness_list=[1], min_cip_count=2, min_interface_count=2,feasibility_checker=feas, preprocessor=pp)
sampler.fit(graphs,grammar_n_jobs=1,grammar_batch_size=1)

