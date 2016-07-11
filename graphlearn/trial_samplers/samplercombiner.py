from graphlearn.utils.monitor import Monitor
from graphlearn.graphlearn import  Sampler
class samplercombiner(object):

    def __init__(self,graphlearn):
        self.parse(graphlearn)

    def getspawns(self, graphlearn):
        '''
        the graphsampler expression is like a tree.
        we turn this into a list and make every graphlearner write into the same monitor object.
        so taht we should be able to run all the sampler sequentially

        we also make graphlearn treat our injected monitor object with respect
        '''
        graphlearn._sample_init_init_monitor=lambda : 0
        def f(s,g):
            s.monitorobject.superscorelist.append(g._score)
            s._score_list.append(g._score)
        funcType = type(graphlearn._score_list_append)
        graphlearn._score_list_append = funcType(f, graphlearn, Sampler)


        li=[graphlearn]
        for gl in graphlearn.__dict__.get('spawn_list',[]):
            li+=self.getspawns(gl)
        return li


    def parse(self,graphlearn):
        self.learners=self.getspawns(graphlearn)
        return self

    def what(self):
        for e in self.learners:
            print e.lsgg.radius_list, e.estimatorobject.inverse_prediction

    def fixmonitor(self, monitor,padlength):
        '''
        sampling may stop at any moment -> needs padding in monitor
        '''
        currentlen= len(monitor.superscorelist)
        if currentlen == 0:
            monitor.superscorelist.append(0)
        monitor.superscorelist+= [monitor.superscorelist[-1]] * (padlength-currentlen)

    def run_single_graph(self,graph,repeats=2):
        # set monitors:
        monitor=Monitor(active=True)
        monitor.superscorelist=[]
        for sampler in self.learners:
            sampler.monitorobject=monitor

        # REPEAT times repeat the samplers in the list
        n_steps=50
        self.steps=n_steps*repeats*len(self.learners)
        print self.steps

        for repeat in range(repeats):
            for samplernum, sampler in enumerate(self.learners):
                samplerresults = sampler.transform(
                    graph_iter=[graph])
                self.fixmonitor(sampler.monitorobject,repeat*len(self.learners)*n_steps+(samplernum)*n_steps)
                graph=samplerresults.next()[0]
        return graph,monitor


    def run_multi_graph(self,graphs,repeats=2):
        for graph in graphs:
            res=self.run_single_graph(graph,repeats=repeats)
            # just hide the failures
            if res != None:
                yield res