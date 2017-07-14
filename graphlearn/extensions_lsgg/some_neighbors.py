import random
import graphlearn.lsgg as base_grammar

class lsgg(base_grammar.lsgg):

    def some_neighbors(self,graph,num_neighbors):

        for n in random.sample(graph.nodes(),len(graph)):
            for graph2 in self._neighbors_given_orig_cips(graph,self._rooted_decompose(graph,n)):
                if num_neighbors > 0:
                    num_neighbors -= 1
                    yield graph2
                else:
                    raise StopIteration
