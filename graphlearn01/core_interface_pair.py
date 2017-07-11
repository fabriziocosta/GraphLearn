'''
datastructure for CoreInterfacePairs.

a cip is a graph, but we collect more information like the hash of the core-part
'''
class CoreInterfacePair:
    """
    this is referred to throughout the code as cip
    it contains the cip-graph and several pieces of information about it.
    """

    def __init__(self,
                 interface_hash=0,
                 core_hash=0,
                 graph=None,
                 radius=0,
                 thickness=0,
                 core_nodes_count=0,
                 count=0,
                 distance_dict={}):
        self.interface_hash = interface_hash
        self.core_hash = core_hash
        self.graph = graph
        self.radius = radius
        self.thickness = thickness
        self.core_nodes_count = core_nodes_count
        self.count = count  # will be used to count how often we see this during grammar creation
        self.distance_dict = distance_dict  # this attribute is slightly questionable. maybe remove it?

    def __str__(self):
        return 'cip: int:%d, cor:%d, rad:%d, thi:%d, rot:%d' % \
               (self.interface_hash, self.core_hash, self.radius, self.thickness, min(self.distance_dict.get(0, [999])))
