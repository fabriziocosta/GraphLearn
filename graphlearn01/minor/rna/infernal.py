'''
here we find items that are needed to get the infernal scores
'''
import subprocess as sp
from graphlearn01.minor.rna import write_fasta, is_sequence
from graphlearn01.graphlearn import Sampler


class AbstractSampler(Sampler):
    def _sample_path_append(self, graph, force=False):
        #self._sample_notes += graph.sequence + "n"
        self._sample_notes += graph._base_graph.graph['sequence'] + "n"
        #super(self.__class__, self)._sample_path_append(graph, force=force) 
        super(AbstractSampler, self)._sample_path_append(graph, force=force)

    def fit_transform(self,eden_sequences):
        self.fit(eden_sequences)
        sequences = [x for y,x in eden_sequences]
        return self.transform(sequences)

def infernal_checker(sequence_list, cmfile='rf00005.cm', cmsearchbinarypath='../toolsdata/cmsearch'):
    '''
    :param sequences: a bunch of rna sequences
    :return: get evaluation from cmsearch
    '''
    write_fasta(sequence_list, filename='temp.fa')
    sequence_list = [s for s in sequence_list if is_sequence(s.replace('F', ''))]
    # print sequence_list
    return call_cm_search(cmfile, 'temp.fa', len(sequence_list), cmsearchbinarypath)


def call_cm_search(cmfile, filename, count, cmsearchbinpath):
    out = sp.check_output('%s -g --noali --incT 0  %s %s' % (cmsearchbinpath, cmfile, filename), shell=True)
    # -g global
    # --noali, we dont want to see the alignment, score is enough
    # --incT 0 we want to see everything with score > 0
    result = {}
    s = out.strip().split('\n')
    for line in s:
        if 'HACK' in line:
            linez = line.split()
            score = float(linez[3]) / 100
            id = int(linez[5][4:])
            result[id] = score
    return [result.get(k, 0) for k in range(count)]
