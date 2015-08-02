import utils.draw as draw
import logging
import networkx as nx
from collections import defaultdict
logger = logging.getLogger(__name__)


def default_check(graph):
    '''
    this is the default feasibility check...
    :param graph:
    :return:
    '''
    # check if graph exists
    if len(graph) < 1:
        logger.log(5,'feasibility got empty graph')
        return False


    if isinstance(graph,nx.DiGraph):
        for node_id in graph.nodes_iter():
            if 'edge' in graph.node[node_id]:
                n=graph.neighbors(node_id)
                n+=graph.predecessors(node_id)
                s=set(n)
                if len(s) != 2:
                    logger.log(5,'feasibility edge check failed')
                    return False


        return True

    # check if all the "edge nodes" have a start and end vertex..
    # if you think edge-node is a oxymoron see "transform edge to vertex" in eden
    for node_id in graph.nodes_iter():
        if 'edge' in graph.node[node_id]:
            if len(graph.neighbors(node_id)) != 2:
                logger.log(5,'feasibility edge check failed')
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

                    draw.graphlearn_draw(graph)
                # and claim unfeasible
                return False
        # no errors found so we are probably good
        return True




def cycles(max_cycle_size):
    return lambda x:not problem_cycle(x,max_cycle_size)


def problem_cycle(graph,max_cycle_size):
    '''

    :param graph:
    :param max_cycle_size:
    :return: problem aru ? True
    '''

    # check for each node:
    for n in graph.nodes_iter():
        if rooted_problem_cycle(graph,n,max_cycle_size):
            return True
    return False

def rooted_problem_cycle(graph,n,max_cycle_size):

    frontier= set([n])
    step=0
    visited=set()

    parent=defaultdict(list)

    while frontier:
        #print frontier
        step+=1

        # give me new nodes:
        next=[]
        for front_node in frontier:
            new = set(graph.neighbors(front_node)) - visited
            next.append(  new )
            for e in new:
                parent[e].append(front_node)


        # we merge the new nodes.   if 2 sets collide, we found a cycle of even length
        while len(next)>1:
            # merge
            s1=next[1]
            s2=next[0]
            merge = s1 | s2

            # check if we hace a cycle
            if  len(merge) < len(s1)+len(s2):
                col= s1&s2
                if root_is_last_common_ancestor(col,parent,n):
                    if step*2 > max_cycle_size:
                        return True
                    return False

            #delete from list
            next[0]=merge
            del next[1]
        next=next[0]



        # now we need to check for cycles of uneven length
        if len(next & frontier) > 0:
            col= next&frontier
            if root_is_last_common_ancestor(col,parent,n):

                if step*2-1 > max_cycle_size:
                    return True
                return False

        # we know that the current frontier didntclose cycles so we dont need to look at them again
        visited = visited | frontier
        frontier=next

    return False

def root_is_last_common_ancestor(col,parent,n):


    # any should be fine. e is closing a cycle
    e= col.pop()
    #print 'r',e

    # need starting positions
    li=parent[e]
    a = [li[0]]
    b = [li[1]]

    #print 'pre',a,b

    a=extpath(a,parent,n)
    b=extpath(b,parent,n)

    #print 'comp',a,b

    return len(set(a) & set(b)) ==0


def extpath(l,parent,n):
    current=l[-1]
    while current != n:
        l.append(  parent[current][0]  )
        current=l[-1]
    return l[:-1]














