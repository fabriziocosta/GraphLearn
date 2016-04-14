import decompose


class GraphTransformer(object):
    def set_param(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, inputs):
        return self

    def fit_transform(self, inputs):
        '''

        Parameters
        ----------
        input : many inputs

        Returns
        -------
        graphwrapper iterator
        '''
        self.fit(inputs)
        return self.transform(inputs)

    def re_transform_single(self, graph):
        '''
        Parameters
        ----------
        graphwrapper

        Returns
        -------
        a postprocessed graphwrapper
        '''
        # mabe a copy?
        return self.transform([graph])[0]

    def transform(self, inputs):
        '''
        Parameters
        ----------
        inputs : list of things

        Returns
        -------
        graphwrapper : iterator
        '''
        return [self.vectorizer._edge_to_vertex_transform(i) for i in inputs]

    def wrap(self, graph):
        raise "OMG OMG OMG somebody tried to call wrap, but we dont wrap anymore."
        return decompose.Decomposer(graph, self.vectorizer)


