#from graphlearn.preservestructure import StructurePreservingGrammar as lsgg
from  graphlearn import structurepreserve as supe

from graphlearn.util import util
import networkx as nx

import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=5)

def test_supe():
    
    # init grpah and grammar 
    g = nx.path_graph(5)
    g = util._edenize_for_testing(g)
    lsgg = supe.StructurePreservingGrammar( thickness=1,
            radii=[0,2,4], preserve_ids=False,
            nodelevel_radius_and_thickness=False)

    # make a cip
    cores = lsgg._get_cores(g)
    cip = lsgg._get_cip(cores[0], g)
    print(cip.ascii())
    
    # train a grammar
    lsgg.fit([g,g,g])
    lsgg.structout()


    # generate neighs
    s = list(lsgg.neighbors(g))
    import structout as so
    so.gprint(s)
    print("done")



