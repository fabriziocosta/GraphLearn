import graph as gt

class PreProcessor(object):

    def fit(self, inputs,vectorizer):
        self.vectorizer=vectorizer
        return self

    def fit_transform(self,inputs,vectorizer):
        '''

        Parameters
        ----------
        input : many inputs

        Returns
        -------
        graphwrapper iterator
        '''
        self.fit(inputs,vectorizer)
        return self.transform(inputs)

    def re_transform_single(self, graphwrapper):
        '''

        Parameters
        ----------
        graphwrapper

        Returns
        -------
        a postprocessed graphwrapper
        '''
        # mabe a copy?
        return gt.Wrapper(graphwrapper, self.vectorizer)

    def transform(self,inputs):
        '''

        Parameters
        ----------
        inputs : list of things

        Returns
        -------
        graphwrapper : iterator
        '''
        return [gt.Wrapper(self.vectorizer._edge_to_vertex_transform(i), self.vectorizer) for i in inputs]


