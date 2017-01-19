from eden.graph import Vectorizer

def node_operation(graph, f):
    # applies function to n,d of nodes(data=True)
    # if you want to do assignments do this: def my(n,d): d[dasd]=asd
    # if you want the result you may use lambda :)

    res=[]
    for n,d in graph.nodes(data=True):
        res.append(f(n,d))
    return res


def prep(graphlist,id=0):
    if not graphlist:
        return {}
    v=Vectorizer()
    map(lambda x: node_operation(x, lambda n, d: d.pop('weight', None)), graphlist)
    csr=v.transform(graphlist)
    hash_function = lambda vec: hash(tuple(vec.data + vec.indices))
    return {hash_function(row): (id,ith) for ith, row in enumerate(csr)}



def intersect():
    pass

def union(glist1,glist2):
    d=prep(glist1,id=1)
    d.update(prep(glist2,id=2))
    return d.values()


def diff(glist0, glist1, list0_id=0, list1_id=1):
    '''
    Parameters
    ----------
    glist1 graphlist
    glist2 graphlist

    Returns
    -------
        graphs not in the intersection
        [(list_id, graph_id)]
    '''
    l0 = prep(glist0,id=list0_id)
    l1 = prep(glist1,id=list1_id)
    res  = [v for k, v in l0.items() if k not in l1]
    res += [v for k, v in l1.items() if k not in l0]
    return res
