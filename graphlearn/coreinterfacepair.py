class CoreInterfacePair:

    """
    this is refered to throughout the code as cip
    it contains the cip-graph and several pieces of information about it.
    """

    def __init__(self, ihash=0, chash=0, graph=0, radius=0, thickness=0, core_nodes_count=0, distance_dict={}):
        self.interface_hash = ihash
        self.core_hash = chash
        self.graph = graph
        self.radius = radius
        self.thickness = thickness
        self.core_nodes_count = core_nodes_count
        self.distance_dict = distance_dict
