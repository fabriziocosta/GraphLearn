import os
import subprocess as sp
import eden
import sklearn
from eden import path
from graphlearn.abstract_graphs.rna import write_fasta, _pairs
from eden.RNA import Vectorizer as EdenRnaVectorizer


'''
consensus folding for rna sequences
'''

class NearestNeighborFolding(object):
    '''
    fit: many structures,  nn model will be build
    transform: returns a structure  , nn are fold together
    '''

    def __init__(self, n_neighbors=4):
        self.n_neighbors = n_neighbors

    def fit(self, sequencelist):
        self.sequencelist = sequencelist
        self.vectorizer = path.Vectorizer(nbits=8)
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
    def __init__(self, n_neighbors=4, structure_mod=False):
        self.n_neighbors = n_neighbors
        self.structure_mod=structure_mod

    def fit(self, sequencelist):
        self.eden_rna_vectorizer = EdenRnaVectorizer(n_neighbors=self.n_neighbors)
        self.eden_rna_vectorizer.fit(sequencelist)

        # after the initial thing: settting min enery high so we never do mfe
        # self.eden_rna_vectorizer.min_energy= -10
        return self

    def transform_single(self, sequence):
        s, neigh = self.eden_rna_vectorizer._compute_neighbors([sequence]).next()
        head, seq, stru, en = self.eden_rna_vectorizer._align_sequence_structure(s, neigh, structure_deletions=True)
        # stru = self._clean_structure(seq,stru) # this is a way to limit the deleted bracket count, idea does not work well
        sequence=sequence[1]
        if self.structure_mod:
            stru,sequence = fix_structure(stru,sequence)
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
    """
    def _pairs(self, struct):
        '''
        Parameters
        ----------
        struct : basestring
        Returns
        -------
        dictionary of ids in the struct, that are bond pairs
        '''
        unpaired = []
        pairs = {}
        for i, c in enumerate(struct):
            if c == '(':
                unpaired.append(i)
            if c == ')':
                partner = unpaired.pop()
                pairs[i] = partner
                pairs[partner] = i
        return pairs
    """


def fix_structure(stru, stri):
    '''
    structure mod is for forgi transformation..
    in forgi, core nodes dont have to be adjacent ->  dont know why currently...
    anyway we fix this by introducing nodes with an F label.

    the problem is to check every (( and )) .
    if the bonding partners are not next to each other we know that we need to act.
    '''
    p = _pairs(stru)
    lastchar = "."
    problems = []
    for i, c in enumerate(stru):
        # checking for )) and ((
        if c == lastchar and c != '.':
            if abs(p[i] - p[i - 1]) != 1:  # the partners are not next to each other
                problems.append(i)
        # )( provlem
        elif c == '(':
            if lastchar == ')':
                problems.append(i)
        lastchar = c
    problems.sort(reverse=True)
    for i in problems:
        stru = stru[:i] + '.' + stru[i:]
        stri = stri[:i] + 'F' + stri[i:]

    return stru, stri