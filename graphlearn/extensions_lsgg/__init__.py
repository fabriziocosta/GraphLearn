

from collections import defaultdict
def _extract_grammar_stats(grammar):
    count_corehashes = defaultdict(int)
    count_interfacehashes = defaultdict(int)
    corecounter = defaultdict(int)
    intercounter = defaultdict(int)
    for ih in grammar.keys():
        for ch in grammar[ih].keys():
            # go over all the combos
            count_corehashes[ch] += 1
            count_interfacehashes[ih] += 1
            count = grammar[ih][ch].count
            corecounter[ch] += count
            intercounter[ih] += count
    return count_corehashes, count_interfacehashes, corecounter, intercounter