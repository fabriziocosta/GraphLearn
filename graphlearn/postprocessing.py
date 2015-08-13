class PostProcessor:
    # this class is your hook if you need to alter graphs created the graphlearn way,

    def __init__(self):
        pass


    def fit(self, sampler):
    # this will also be called by sampler... to be able to adjust this postprocessor
        pass

    # postprocess will be called by graphlearn. so put whatever you want to do here :)
    def postprocess(self, graph):
        return graph
