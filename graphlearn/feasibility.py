
import utils.draw as draw
import logging
logger = logging.getLogger(__name__)

def default_check(graph):
    if len(graph) < 1:
        logger.debug('ERROR: empty graph')
        return False
    for node_id in graph.nodes_iter():
        if 'edge' in graph.node[node_id]:
            if len(graph.neighbors(node_id)) != 2:
                logger.debug('ERROR: feasibility edge check failed, (interface twist phenomenon probably)')
                return False
    return True

class FeasibilityChecker():

    def __init__(self, draw_problem=False):

        self.checklist = []
        self.checklist.append(default_check)
        self.draw_problem = draw_problem

    def check(self, graph):
        for f in self.checklist:
            if f(graph) == False:
                if self.draw_problem:
                    draw.display(graph)
                return False
        return True