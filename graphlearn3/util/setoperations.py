
import copy




def difference(grammar, other_grammar, substract_cip_count=False):
    """difference between grammars"""
    grammar = copy.deepcopy(grammar)
    for interface in list(grammar.productions):
        if interface in other_grammar.productions:
            for core in list(grammar.productions[interface]):
                if core in other_grammar.productions[interface].keys():
                    grammar.productions[interface].pop(core)
        if len(grammar.productions[interface]) < 2:
            grammar.productions.pop(interface)
    return grammar


def union(grammar, other_grammar):
    """union of grammars"""
    grammar = copy.deepcopy(grammar)
    for interface in list(grammar.productions):
        if interface in other_grammar.productions:
            for core in list(grammar.productions[interface]):
                if core not in other_grammar.productions[interface]:
                    grammar.productions[interface][core] = other_grammar.productions[interface][core]
        else:
            grammar.productions[interface] = copy.deepcopy(other_grammar[interface])
    return grammar

def intersect(grammar, other_grammar):
    """intersection of grammars"""
    grammar = copy.deepcopy(grammar)
    for interface in list(grammar.productions.keys()):
        if interface in other_grammar.productions:
            for core in list(grammar.productions[interface].keys()):
                if core not in other_grammar.productions[interface]:
                    grammar.productions[interface].pop(core)
            if len(grammar.productions[interface]) < 2:
                grammar.productions.pop(interface)
        else:
            grammar.productions.pop(interface)
    return grammar
