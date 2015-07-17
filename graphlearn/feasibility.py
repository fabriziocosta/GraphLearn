import utils.draw as draw
import logging
import networkx as nx

logger = logging.getLogger(__name__)


def default_check(graph):
    '''
    this is the default feasibility check...
    :param graph:
    :return:
    '''
    # check if graph exists
    if len(graph) < 1:
        logger.debug('ERROR: empty graph')
        return False


    if isinstance(graph,nx.DiGraph):
        return True

    # check if all the "edge nodes" have a start and end vertex..
    # if you think edge-node is a oxymoron see "transform edge to vertex" in eden
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
        # for all the check functions
        for f in self.checklist:
            # if it fails:
            if f(graph) is False:
                # we may draw the graph
                if self.draw_problem and len(graph) > 0:

                    draw.display(graph)
                # and claim unfeasible
                return False
        # no errors found so we are probably good
        return True
