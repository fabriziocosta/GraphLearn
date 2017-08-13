'''
sometimes input graphs or graphs after a substitution need a
little fixing to be able to be used by graphlearn.

graphtransformer: graph -> object the decomposer unterstands how to use
'''

from eden.graph  import   _edge_to_vertex_transform

class GraphTransformer(object):


    def fit(self, inputs):
        return self

    def fit_transform(self, inputs,negstuff=[]):
        '''

        Parameters
        ----------
        input : many inputs

        Returns
        -------
        graphwrapper iterator
        '''
        inputs=list(inputs)
        self.fit(inputs)
        return self.transform(inputs+negstuff)

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
        return [_edge_to_vertex_transform(i) for i in inputs]



