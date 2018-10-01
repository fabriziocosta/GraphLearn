
import copy




def difference(grammar, other_grammar, substract_cip_count=False):
    """difference between grammars"""
    grammar = copy.deepcopy(grammar)
    for interface in grammar.productions:
        if interface in other_grammar:
            for core in grammar.productions[interface]:
                if core in other_grammar[interface].keys():
                    grammar.productions[interface].pop(core)


def union(grammar, other_grammar):
    """union of grammars"""
    grammar = copy.deepcopy(grammar)
    for interface in grammar.productions:
        if interface in other_grammar:
            for core in grammar.productions[interface]:
                if core not in other_grammar[interface]:
                    grammar.productions[interface][core] = other_grammar[interface][core]
        else:
            grammar.productions[interface] = copy.deepcopy(other_grammar[interface])

def intersect(grammar, other_grammar):
    """intersection of grammars"""
    grammar = copy.deepcopy(grammar)
    for interface in grammar.productions.keys():
        if interface in other_grammar:
            for core in grammar.productions[interface].keys():
                if core not in other_grammar[interface]:
                    grammar.productions[interface].pop(core)
        else:
            grammar.productions.pop(interface)
