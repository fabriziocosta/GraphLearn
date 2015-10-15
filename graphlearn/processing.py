import graph as gt

class PreProcessor(object):


    def set_param(self,vectorizer):
        self.vectorizer=vectorizer

    def fit(self, inputs):
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
        self.fit(inputs)
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





class PostProcessor(object):

    def fit(self, preprocessor):
        self.pp=preprocessor
        return self

    def fit_transform(self,preprocessor,inputs):
        self.fit(preprocessor)
        return self.transform(inputs)

    def re_transform_single(self, input):
        return self.transform([input])[0]

    def transform(self,inputs):
        return self.pp.transform(inputs)


