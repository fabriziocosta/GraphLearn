import copy
from collections import defaultdict,namedtuple
from networkx.algorithms import isomorphism as iso
import scipy
import joblib


import myutils
import utils.myeden as myutils
import utils.draw as mydraw

import extract
import eden.hasher as hasher
from eden.graph import Vectorizer
import networkx as nx
from eden.util import fit_estimator,fit
import itertools
import random
import math
from multiprocessing import Pool
import numpy as np
from multiprocessing import Manager


'''
    the adaptiveMHgraphsamper has 3 parts:
        -init()  will be followed by
            -load() 
            -train_estimator_and_extract_grammar()
        
        -core substitution 
            in this section you can find substitution related function
            these are of no interest to you
        
        -mass_improve_random(self,graph_iter,times=20)
            will improve a set of graphs 'times' times each and 
            yield when done        
    
    then there is the grammar functions class to induce grammars
        probably not so interesting
    
    and we have a feasibility checker to see if generated graphs
    are valid
        may be interesting to the user
    
'''


                
                
class adaptiveMHgraphsampler:
    '''
         wirte something
    '''

    def __init__(self):
        self.bitmask = pow(2,20) -1
        self.feasibility_checker=feasibility_checker()
        
    def save(self,filename):
        joblib.dump(self.__dict__,filename,compress=1 )

    def load(self,filename):
        self.__dict__=joblib.load(filename)

    def induce_grammar(self,G_iterator,radius_list, thickness_list,n_jobs=4):
        self.substitute_grammar= {}
        
        self.grammar_functions=local_substitutable_graph_grammar(self.substitute_grammar,
                    radius_list,thickness_list)
                    
        self.grammar_functions.readgraphs(G_iterator,n_jobs)
        self.grammar_functions.clean()
        
    
    def train_estimator(self,iterable_pos_train, n_jobs=-1, cv=10):
        # i think this works on normal graphs..
        vectorizer=Vectorizer( complexity=3 )
        X_pos_train = vectorizer.transform( iterable_pos_train, n_jobs=n_jobs )
        X_neg_train = X_pos_train.multiply(-1)
        #optimize hyperparameters classifier
        self.estimator = myutils.my_fit_estimator(positive_data_matrix=X_pos_train, negative_data_matrix=X_neg_train, cv=cv, n_jobs=n_jobs)
        l=[self.estimator.decision_function(g) for g in X_pos_train]
        l.sort()
        element=int(len(l)*.1)
        self.estimator.intercept_ -= l[element]
        return self.estimator
    

    def train_estimator_and_extract_grammar(self,G_pos,radius_list,thickness_list,n_jobs=-1):
        G_iterator,G_pos = itertools.tee(G_pos)
        self.induce_grammar(G_iterator,radius_list,thickness_list,n_jobs)
        self.train_estimator(G_pos,n_jobs)
        
        
    
    ###########################  core substitution things ####################
    
    def merge(self,G,node,node2): 
        '''
        all nodes are strings, node is the king
        '''
        for n in G.neighbors(node2):
            G.add_edge(node,n)
        G.node[node]['interface']=True
        G.remove_node(node2)


    def find_isomorphism(self,home,other):
        matcher= lambda x,y: x['label']==y['label'] 
        GM=iso.GraphMatcher(home,other,node_match=matcher)
        if GM.is_isomorphic()== False:
            return {}
        return GM.mapping


    def core_substitution(self,graph, subgraph, nkgraph):
        '''
        graph is the whole graph..
        subgraph is the interfaceregrion in that we will transplant
        nkgraph which is the interface and the new core
        '''
        nocore=[ n for n,d in nkgraph.nodes(data=True) if d.has_key('core')==False ]
        nksub=nx.subgraph(nkgraph,nocore)

        nocore=[ n for n,d in subgraph.nodes(data=True) if d.has_key('core')==False ]
        subgraph_nocore=nx.subgraph(subgraph,nocore)

        iso=self.find_isomorphism( subgraph_nocore, nksub)
        # i think this happens if no iso was found, untestet
        if len(iso) != len(subgraph_nocore):
            return nx.Graph()


        G=nx.union(graph,nkgraph,rename=('','-'))
        #removing old core  .,., moved this here replaced graph.remove_node(n) with G.... 
        nocore=[ n for n,d in subgraph.nodes(data=True) if d.has_key('core') ]
        for n in nocore:
            G.remove_node(str(n))

        for k,v in iso.iteritems():
            self.merge(G,str(k),'-'+str(v))
            
        #renaming the core nodes .. neccessary?
        #for n,d in G.nodes(data=True):
        #    d['label']=n
        G=nx.convert_node_labels_to_integers(G)
        
        return G


    ############ imporoving stuff ##################
    
    def get_random_candidate(self,graph):
        '''
        graph is now expanded
        '''
        lsgg=self.grammar_functions
        #old_expanded=myutils.expand_edges([graph]).next()

        for tries in range(20):
            candidate,radius,thickness = self.choose_random_cores_and_interfaces(graph,lsgg.radius_list,lsgg.thickness_list)
            if len(candidate) < 1:
                continue
            candidate=candidate[0]
            #cid = core interface data class :) 
            core_cid_dict= self.substitute_grammar.get(candidate.interface_hash,{})
            
            if len(core_cid_dict) == 0:
                continue
                
            else:
                for core_hash in core_cid_dict.keys():
                    if 'sameradius' in self.improvement_rules:
                        if core_cid_dict[core_hash].radius != radius:
                            continue
                    substitution_hash = candidate.core_hash ^ core_hash ^ candidate.interface_hash
                    if  substitution_hash in graph.tried_and_failed_substitutions:
                        continue
                    newgraph=core_cid_dict[core_hash].graph
                    ng=self.core_substitution(graph,candidate.graph,newgraph)
                    graph.tried_and_failed_substitutions.append( substitution_hash )
                    if len(ng) > 1:
                        return ng
                    
        return nx.Graph()
    
    def choose_random_cores_and_interfaces(self,graph,radius_list,thickness_list):
        node= random.choice(graph.nodes())
        if 'edge' in graph.node[node]:
            node= random.choice( graph.neighbors(node) )
            
        #random radius and thickness
        radius= random.choice( radius_list )
        thickness= random.choice( thickness_list )
        
        core_interface_data_list=extract.extract_core_and_interface(node,graph,[radius],[thickness])
        if len(core_interface_data_list)>0:
            return core_interface_data_list,radius,thickness
        return [],0,0
    
    
    
    def mass_improve_random(self,graph_iter,times=20,n_jobs=0,improvement_rules={'sameradius':1} ):
        '''
        improve every graph in the graph iter times times!
        '''
        self.improvement_rules=improvement_rules
        
        manager = Manager()
        shelve=manager.dict()
        for k,v in self.substitute_grammar.iteritems():
            shelve[k]=v
        self.substitute_grammar=shelve
        
        if n_jobs==0:
            for graph in graph_iter:
                yield self.improve_loop(graph,times)
        else:
            problems=itertools.izip(
                graph_iter,itertools.repeat(times),itertools.repeat(self))
            pool=Pool(processes=n_jobs)
            
            it=pool.imap_unordered(improve_loop_multi,problems,1)
            for liste in it:
                yield liste
                   
    
    def improve_loop(self,graph,times=20):
        scores=[]
        graph=myutils.expand_edges(graph)
        graph=extract.preprocess(graph)
        score=-3
        g0=nx.Graph(graph)
        vectorizer = Vectorizer(complexity =3 )
        sim=-1
        if 'similarity' in self.improvement_rules:
            sim=self.improvement_rules['similarity']
            
            ### THERE IS A BUG HERE NEED FIX !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
            ### add similarity:0,0 to mass imp random to trigger
            vectorizer._reference_vec=vectorizer._convert_dict_to_sparse_matrix(vectorizer._transform(0, g0))
        

        for x in xrange(times):
            if sim != -1:
                if vectorizer._similarity([graph],weights=[1]) < sim:
                    scores+=[score]* (times-x)
                    break
                    
            graph,score=self.improve_random(graph,score)
            scores.append(score)
            
        graph.scorehistory=scores
        
        return graph

    def improve_random(self,graph,oldscore,debug=0,powerbonus=0.0,cooling=0.0,):
        '''
        graph is now expanded
        debug will remove core/interface attributes of nodes
            this is not neccessary but makes the graph 
            prettier to print
        '''
        
        graph.tried_and_failed_substitutions=[]
        
        if debug > 0:
            for n,d in graph.nodes(data=True):
                d.pop('core',None)
                d.pop('interface',None)

        
        candidate=self.get_random_candidate(graph)
        if len(candidate)==0:
            if debug > 1:
                print "no candidate"
            return graph,oldscore
        vectorizer = myutils.my_vectorizer( complexity=3 )
       
        graph2=candidate
        #mydraw.display(graph2,contract=False)
        try:
            transformed_graph=vectorizer.transform2(graph2)   
        except:
            if debug > 1:
                print "transformation failed"
            return graph,oldscore
        value=self.estimator.decision_function(transformed_graph)[0]
        
        #self.feasibility_checker.check(graph2,value,graph,oldscore)
        
        kfactor=15
        v1=1.0/(1+math.exp(-oldscore*kfactor))
        v2=1.0/(1+math.exp(-value*kfactor))
        randf= random.random()
        
        #print v2/v1,v1,v2,randf,value,oldscore
        
        if v2/v1 > randf:
            return graph2,value
        return graph,oldscore


#ok moving this here instead of leaving it where it belongs prevents pickling errar .. 
#dont quite get it ...        
def improve_loop_multi(x):
    return x[2].improve_loop(x[0],x[1])        

################ALL THE THINGS HERE SERVE TO LEARN A GRAMMAR ############
   
class subgraphdatac:
    def __init__(self):
        self.count=0
        
class local_substitutable_graph_grammar:
    '''
    i had this class inherit from default dict, but that breaks joblib oOo
    and i cant load anymore.
    '''
    # move all the things here that are needed to extract grammar
    def __init__(self,ddict,radius_list,thickness_list,core_interface_pair_remove_threshold=3,interface_remove_threshold=2):
        self.data=ddict
        self.interface_remove_threshold= interface_remove_threshold
        self.radius_list=radius_list
        self.thickness_list=thickness_list
        self.core_interface_pair_remove_threshold=core_interface_pair_remove_threshold



    def readgraphs(self,graphs,n_jobs=4):
        if n_jobs==1:
            self.readgraphs_single(graphs)
        else:
            self.readgraphs_multi(graphs,n_jobs)
    
    def grammar_add_core_interface_data(self,cid):
        if cid.interface_hash not in self.data:
            self.data[cid.interface_hash]={}
        
        subgraphdata = subgraphdatac()
        if cid.core_hash in self.data[cid.interface_hash]:
            subgraphdata=self.data[cid.interface_hash][cid.core_hash]
        else:
            self.data[cid.interface_hash][cid.core_hash]=subgraphdata
        subgraphdata.count+=1
        if subgraphdata.count == self.core_interface_pair_remove_threshold:  
            subgraphdata.graph=cid.graph
            subgraphdata.radius=cid.radius
            subgraphdata.thickness=cid.thickness
    
    def readgraphs_single(self,graphs):
        for gr in graphs:
            for core_interface_data_list in extract_cores_and_interfaces(gr,self.radius_list,self.thickness_list):
                for cid in core_interface_data_list:
                    self.grammar_add_core_interface_data(cid)
                    
    
    def readgraphs_multi(self,graphs,n_jobs):
        problems=itertools.izip(graphs, itertools.repeat(self.radius_list),itertools.repeat(self.thickness_list) )
        pool=Pool(processes=n_jobs)
        it=pool.imap_unordered(extract_cores_and_interfaces_multi,problems,10)
        for core_interface_data_listlist in it:
            for core_interface_data_list in core_interface_data_listlist:
                for cid in core_interface_data_list:
                    self.grammar_add_core_interface_data(cid)
                    
            


   # rename to trim // pass trimvalue
    def clean(self):
        for interface in self.data.keys():
            for core in self.data[interface].keys():
                if self.data[interface][core].count<self.core_interface_pair_remove_threshold:
                    self.data[interface].pop(core)
            if len(self.data[interface]) < self.interface_remove_threshold:
                self.data.pop(interface)

def extract_cores_and_interfaces(graph,radius_list,thickness_list):
    #expand the graph 
    graph=myutils.expand_edges(graph)
    graph=extract.preprocess(graph)
    for node in graph.nodes_iter():
        if 'edge' in graph.node[node]:
            continue
        core_interface_list=extract.extract_core_and_interface(node,graph,radius_list,thickness_list)
        if len(core_interface_list)>0:
            yield core_interface_list



def extract_cores_and_interfaces_multi(x):
    #expand the graph 
    graph,radius_list,thickness_list = x
    graph=myutils.expand_edges(graph)
    graph=extract.preprocess(graph)
    ret= []
    for node in graph.nodes_iter():
        if 'edge' in graph.node[node]:
            continue
        core_interface_list=extract.extract_core_and_interface(node,graph,radius_list,thickness_list)
        if len(core_interface_list)>0:
            ret.append(core_interface_list)
    return ret
    


################################ FEASIBILITY CHECKER ###################

   
class feasibility_checker():
    def __init__(self):
        self.checklist=[]
        self.checklist.append(defaultcheck)
           
    def check(self,ng,ns,og,os):
        for f in self.checklist:
            if f(ng,ns,og,os)==False:
                return False
        return True
           
                
def defaultcheck(ng,ns,og,os):
    for node_id in ng.node_iter():
        if 'edge' in ng.node[node_id]:
            if len(ng.neighbors(node_id)) != 2:
                return False
    return True



