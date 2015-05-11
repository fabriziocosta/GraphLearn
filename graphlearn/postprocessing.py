


import logging


class PostProcessor:

    def __init__(self):
        # careful here, i got an error in graphlearn.save when this was active:
        #self.logger=logging.getLogger('log')
        pass

    def postprocess(self, graph):
        return graph

