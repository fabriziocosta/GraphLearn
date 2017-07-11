'''
feasibility checker for graphs druing sampling.
easily extendable by custom checks
'''

import utils.draw as draw
import logging
import networkx as nx

# from collections import defaultdict
logger = logging.getLogger(__name__)


def default_check(graph):
    '''
    this is the default feasibility check...
    '''
    # check if graph exists
    if len(graph) < 1:
        logger.log(5, 'feasibility got empty graph')
        return False

    if isinstance(graph, nx.DiGraph):
        for node_id in graph.nodes_iter():
            if 'edge' in graph.node[node_id]:
                n = graph.neighbors(node_id)
                n += graph.predecessors(node_id)
                s = set(n)
                if len(s) != 2:
                    logger.log(5, 'feasibility edge check failed')
                    return False

        return True

    # check if all the "edge nodes" have a start and end vertex..
    # if you think edge-node is a oxymoron see "transform edge to vertex" in eden
    for node_id in graph.nodes_iter():
        if 'edge' in graph.node[node_id]:
            if len(graph.neighbors(node_id)) != 2:
                logger.log(5, 'feasibility edge check failed')
                return False
    return True


class FeasibilityChecker():
    def __init__(self, checklist=[default_check], draw_problem=False):
        self.checklist = checklist
        self.draw_problem = draw_problem

    def check(self, graph):
        # for all the check functions
        for f in self.checklist:
            # if it fails:
            if f(graph) is False:
                # we may draw the graph
                if self.draw_problem and len(graph) > 0:
                    print 'feasibility failed: drawing graph'
                    draw.graphlearn(graph, size=10)
                # and claim unfeasible
                return False
        # no errors found so we are probably good
        return True


def cycle_feasibility_checker(max_cycle_size):
    max_cycle_size *= 2
    from utils import cycles
    return FeasibilityChecker(checklist=[default_check, cycles.cycles(max_cycle_size)])
