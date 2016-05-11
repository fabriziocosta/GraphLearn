'''
tools for cycles in graphs.

'''

from collections import defaultdict

# for the checking we need a function that takes a graph.. so we make one.. :)
def cycles(max_cycle_size):
    return lambda x: not problem_cycle(x, max_cycle_size)


def problem_cycle(graph, max_cycle_size):
    '''
    for each node we check if a questionable cycle exists
    :param graph:
    :param max_cycle_size:
    :return: problem aru ? True
    '''
    for n in graph.nodes_iter():
        if rooted_problem_cycle(graph, n, max_cycle_size):
            return True
    return False


def rooted_problem_cycle(graph, n, max_cycle_size):
    """
    :param graph:
    :param n: start node
    :param max_cycle_size:
    :return: do we have a questionable cycle that includes n



    so we start in node n,
    then we expand 1 node further in each step.
    if we meet a node we had before we found a cycle.

    there are 3 possible cases.
        - frontier hits frontier -> cycle of even length
        - frontier hits visited nodes -> cycle of uneven length
        - it is also possible that the newly found cycle doesnt contain our start node. so we check for that
    """

    frontier = set([n])
    step = 0
    visited = set()
    parent = defaultdict(list)

    while frontier:
        # print frontier
        step += 1

        # give me new nodes:
        next = []
        for front_node in frontier:
            new = set(graph.neighbors(front_node)) - visited
            next.append(new)
            for e in new:
                parent[e].append(front_node)

        # we merge the new nodes.   if 2 sets collide, we found a cycle of even length
        while len(next) > 1:
            # merge
            s1 = next[1]
            s2 = next[0]
            merge = s1 | s2

            # check if we havee a cycle   => s1,s2 overlap
            if len(merge) < len(s1) + len(s2):
                col = s1 & s2
                if root_is_last_common_ancestor(col, parent, n):
                    if step * 2 > max_cycle_size:
                        return True
                    return False

            # delete from list
            next[0] = merge
            del next[1]
        next = next[0]

        # now we need to check for cycles of uneven length => the new nodes hit the old frontier
        if len(next & frontier) > 0:
            col = next & frontier
            if root_is_last_common_ancestor(col, parent, n):

                if step * 2 - 1 > max_cycle_size:
                    return True
                return False

        # we know that the current frontier didntclose cycles so we dont need to look at them again
        visited = visited | frontier
        frontier = next

    return False


def root_is_last_common_ancestor(col, parent, n):
    # any should be fine. e is closing a cycle,
    # note: e might contain more than one hit but we dont care
    e = col.pop()
    # print 'r',e
    # we closed a cycle on e so e has 2 parents...
    li = parent[e]
    a = [li[0]]
    b = [li[1]]
    # print 'pre',a,b
    # get the path until the root node
    a = extend_path_to_root(a, parent, n)
    b = extend_path_to_root(b, parent, n)
    # print 'comp',a,b
    # of the paths to the root node dont overlap, the root node musst be in the loop
    return len(set(a) & set(b)) == 0


def extend_path_to_root(l, parent, n):
    """
    :param l: list with start node
    :param parent: the tree like dictionary that contains each nodes parent(s)
    :param n: root node. probably we dont really need this since the root node is the orphan
    :return: path from l to n , be moving up the tree. note that we dont care to get the shortest path.
    """
    current = l[-1]
    while current != n:
        l.append(parent[current][0])
        current = l[-1]
    return l[:-1]
