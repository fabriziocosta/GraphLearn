


# this file is here to hide the uglyness from the notebooks

def setparameters(sampler):
    '''
    ok,lets set all the options just to make sure we didnt miss any
    '''

    # note, i just copy pasted  sample of graphlearn.sampler..
    # this way we are flexible and dont have to think if future changes are incoming
    probabilistic_core_choice=True
    score_core_choice=False
    max_core_size_diff=-1
    similarity=-1
    n_samples=None
    proposal_probability=False
    batch_size=10
    target_orig_cip=False
    quick_skip_orig_cip=False
    select_cip_max_tries=20
    burnin=0
    generator_mode=False
    omit_seed=True
    keep_duplicates=False

    sampler.proposal_probability = proposal_probability
    sampler.similarity = similarity

    if probabilistic_core_choice + score_core_choice + max_core_size_diff == -1 > 1:
        raise Exception('choose max one cip choice strategy')

    #if n_samples:
    #    sampler.sampling_interval = int((n_steps - burnin) / (n_samples + omit_seed - 1))
    #else:
    #    sampler.sampling_interval = 9999


    sampler.quick_skip_orig_cip = quick_skip_orig_cip
    sampler.target_orig_cip = target_orig_cip

    # the user doesnt know about edge nodes.. so this needs to be done
    max_core_size_diff = max_core_size_diff * 2
    sampler.max_core_size_diff = max_core_size_diff
    sampler.select_cip_max_tries = select_cip_max_tries
    sampler.burnin = burnin
    sampler.omit_seed = omit_seed
    sampler.batch_size = batch_size
    sampler.probabilistic_core_choice = probabilistic_core_choice
    sampler.score_core_choice = score_core_choice
    sampler.monitor=False
    sampler.monitors=[]
    sampler.maxbacktrack=0
    sampler.keep_duplicates = keep_duplicates
    # adapt grammar to task:
    sampler.lsgg.preprocessing(4,
                            max_core_size_diff,
                            probabilistic_core_choice )
    sampler.step=3 # whatever





# for the easy mode:
def easy_get_new_graphs(graphwrap,sampler):
    res=[]
    graphwrap.clean()
    res= [sampler._propose(graphwrap) for i in range(8)]

    for i,gw in enumerate(res):
        gw._base_graph.graph['info'] = str(i)
    #for i in range(8):
    #    gr2=      nx.Graph(gr)
    #    cip =     sampler.select_original_cip(gr2)
    #    newcip =  sampler._select_cips(cip).next()
    #    newgr=    graphtools.core_substitution(gr2, cip.graph, newcip.graph)
    #    res.append(newgr)

    return res


# used in non easy mode

def getargz(sampler):
    return{'radius_list':sampler.radius_list,
        'thickness_list':sampler.thickness_list,
        'hash_bitmask':sampler.hash_bitmask,
        #'filter':sampler.node_entity_check
        }


def get_cips(graphman,sampler,root,d):
    cips = graphman.rooted_core_interface_pairs(root,**d)
    res=[]
    counter=0
    for cip in cips:
        if cip.interface_hash in sampler.lsgg.productions:
            new_cips=sampler.lsgg.productions[cip.interface_hash].values()
            for nc in new_cips:
                # save the original_cip_graph for replacement later
                nc.orig=cip.graph
                nc.graph.graph['info']=str(counter)
                counter+=1
            res+=new_cips

    return res





