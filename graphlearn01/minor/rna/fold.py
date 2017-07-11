'''
consensus folding for rna sequences

NearestNeighborFolding is using mlocarna
EdenNNF is aligning and then folding
'''

import os
import subprocess as sp
import sklearn
from eden import sequence as eden_sequence
from eden_rna import RNAFolder
from graphlearn01.minor.rna import write_fasta, _pairs


class NearestNeighborFolding(object):
    '''
    fit: many structures,  nn model will be build
    transform: returns a structure  , nn are fold together
    '''

    def __init__(self, n_neighbors=4):
        self.n_neighbors = n_neighbors

    def fit(self, sequencelist):
        self.sequencelist = sequencelist
        self.vectorizer = eden_sequence.Vectorizer(nbits=8)
        data_matrix = self.vectorizer.transform(self.sequencelist)
        self.neigh = sklearn.neighbors.LSHForest()
        # self.neigh =sklearn.neighbors.NearestNeighbors()

        self.neigh.fit(data_matrix)
        return self

    def get_nearest_sequences(self, sequence):
        needle = self.vectorizer.transform([sequence])
        neighbors = self.neigh.kneighbors(needle, n_neighbors=self.n_neighbors)[1][0].tolist()
        return [self.sequencelist[i] for i in neighbors]

    def transform(self, sequences):
        for seq in sequences:
            yield self.transform_single(seq)

    def transform_single(self, sequence):
        seqs = self.get_nearest_sequences(sequence)
        seqs.append(sequence)
        filename = './tmp/fold' + str(os.getpid())
        write_fasta(seqs, filename=filename)
        try:
            r = self.call_folder(filename=filename)
        except:
            print sequence, seqs
        return r

    def call_folder(self, filename='NNTMP', id_of_interest=None):
        if id_of_interest is None:
            id_of_interest = self.n_neighbors
        try:
            out = sp.check_output('mlocarna %s | grep "HACK%d\|alifold"' % (filename, id_of_interest),
                                  shell=True)
            out = out.split('\n')
            seq = out[0].split()[1]
            stru = out[1].split()[1]
            stru = list(stru)
        # stru2=''.join(stru)
        except:
            print 'folding problem:', out, filename, id_of_interest

        # find  deletions
        ids = []
        for i, c in enumerate(seq):
            if c == '-':
                ids.append(i)

        # take care of deletions
        # remove brackets that dont have a partner anymore
        pairdict = _pairs(stru)
        for i in ids:
            if stru[i] != '.':
                stru[pairdict[i]] = '.'

        # delete
        ids.reverse()
        for i in ids:
            del stru[i]
            # stru=stru[:i]+stru[i+1:]

        # print seq
        # print stru2
        # print ''.join(stru)

        return ''.join(stru)


class EdenNNF(NearestNeighborFolding):
    def __init__(self, n_neighbors=4):
        self.n_neighbors = n_neighbors


    def fit(self, eden_sequences):
        self.eden_rna_vectorizer = RNAFolder.Vectorizer(n_neighbors=self.n_neighbors)
        self.eden_rna_vectorizer.fit(eden_sequences)

        # after the initial thing: settting min enery high so we never do mfe
        # self.eden_rna_vectorizer.min_energy= -10
        return self

    def transform_single(self, sequence):
        s, neigh = self.eden_rna_vectorizer._compute_neighbors([sequence]).next()
        # s is a string and neigh is a list of tupels
        head, seq, stru, en = self.eden_rna_vectorizer._align_sequence_structure(('veryUniqueSequencename',s), neigh, structure_deletions=True)
        # stru = self._clean_structure(seq,stru) # this is a way to limit the deleted bracket count, idea does not work well

        return stru, en, sequence

    def _clean_structure(self, seq, stru):
        '''
        Parameters
        ----------
        seq : basestring
            rna sequence
        stru : basestring
            dotbracket string
        Returns
        -------
        the structure given may not respect deletions in the sequence.
        we transform the structure to one that does
        '''
        DELETED_BRACKETS = 0

        # find  deletions in sequence
        ids = []
        for i, c in enumerate(seq):
            if c == '-':
                ids.append(i)
        # remove brackets that dont have a partner anymore
        stru = list(stru)
        pairdict = _pairs(stru)
        for i in ids:
            stru[pairdict[i]] = '.'
            DELETED_BRACKETS += 1
        # delete deletions in structure
        ids.reverse()
        for i in ids:
            del stru[i]
        stru = ''.join(stru)

        if "(())" in stru:
            DELETED_BRACKETS += 4
        if "(..)" in stru:
            DELETED_BRACKETS += 2
        if "(.)" in stru:
            DELETED_BRACKETS += 2
        # removing obvious mistakes
        stru = stru.replace("(())", "....")
        stru = stru.replace("(.)", "...")
        stru = stru.replace("(..)", "....")

        if DELETED_BRACKETS > 4:
            return None
        return stru

