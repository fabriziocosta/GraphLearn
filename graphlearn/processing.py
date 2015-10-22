import graph as gt

class PreProcessor(object):
    def set_param(self,vectorizer):
        self.vectorizer=vectorizer

    def fit(self, inputs):
        return self

    def fit_transform(self,inputs):
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
        return self.wrap(graphwrapper)

    def transform(self,inputs):
        '''
        Parameters
        ----------
        inputs : list of things

        Returns
        -------
        graphwrapper : iterator
        '''
        return [self.wrap(self.vectorizer._edge_to_vertex_transform(i)) for i in inputs]

    def wrap(self,graph):
        return gt.Wrapper(graph,self.vectorizer)



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


