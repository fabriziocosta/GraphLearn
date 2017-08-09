import random
## SELECTING A CIP FROM THE CURRENT GRAPH
def select_original_cip( decomposer, sampler):
    """
    selects a cip from the original graph.
    (we try maxtries times to make sure we get something nice)

    - original_cip_extraction  takes care of extracting a cip
    - accept_original_cip makes sure that the cip we got is indeed in the grammar
    """
    if sampler.orig_cip_score_tricks:
        decomposer.mark_median(out='is_good', estimator=sampler.estimatorobject.estimator, vectorizer=sampler.vectorizer)

    # draw.graphlearn(graphman.abstract_graph(), size=10)
    # draw.graphlearn(graphman._abstract_graph, size=10)
    # print graphman

    failcount_score = 0
    failcount_grammar = 0
    nocip = 0
    for x in range(sampler.select_cip_max_tries):
        # exteract_core_and_interface will return a list of results,
        # we expect just one so we unpack with [0]
        # in addition the selection might fail because it is not possible
        # to extract at the desired radius/thicknes
        cip = _get_original_cip(decomposer,grammar=sampler.lsgg)
        if not cip:
            nocip += 1
            continue
        cip = cip[0]

        # print cip
        grammar_ok, score_ok= _accept_original_cip(cip,grammar=sampler.lsgg,orig_cip_max_positives=sampler.orig_cip_max_positives,
                                     orig_cip_min_positives=sampler.orig_cip_min_positives
                                     ,orig_cip_score_tricks=sampler.orig_cip_score_tricks, sampler=sampler)
        if grammar_ok and score_ok:
            yield cip
        else:
            failcount_grammar += not grammar_ok
            failcount_score += not score_ok

    sampler._samplelog(
            'cip_select select orig cip failed; obtained %d cips in %d tries, \
of which: bad_score:%d, not_in_grammar:%d '
            % (sampler.select_cip_max_tries-nocip,sampler.select_cip_max_tries, failcount_score,failcount_grammar),level=5)

    #from utils import draw
    #draw.debug(decomposer._base_graph, label="label")


def _get_original_cip( decomposer,grammar=None):
    '''
    selects a cip to alter in the graph.

    Parameters
    ----------
    decomposer

    Returns
    -------
        a random cip from decomposer

    USED ONLY IN SELECT_ORIGINAL_CIP
    '''
    return decomposer.random_core_interface_pair(radius_list=grammar.radius_list, thickness_list=grammar.thickness_list)

def _accept_original_cip( cip,grammar=None,
                         orig_cip_max_positives=None,
                         orig_cip_min_positives=None,
                         orig_cip_score_tricks=None,sampler=None):
    """

    see if the choosen cip in the original is "ok"

    Parameters
    ----------
    cip: the cip we need to judge

    Returns
    -------
    good or nogood (bool)
    """
    score_ok = True
    if orig_cip_score_tricks:
        imp = []
        for n, d in cip.graph.nodes(data=True):
            if 'interface' not in d and 'edge' not in d:
                imp.append(d['is_good'])


        positives=(float(sum(imp)) / len(imp))

        if not (orig_cip_min_positives <= positives <= orig_cip_max_positives ):
            score_ok=False
            #sampler._samplelog(  'cip_select orig: scores_ok: %.4f %.4f %.4f' % (orig_cip_min_positives,positives,orig_cip_max_positives))


        #if (float(sum(imp)) / len(imp)) > orig_cip_max_positives:
        #    score_ok = False
        #if (float(sum(imp)) / len(imp)) < orig_cip_min_positives:
        #    score_ok = False
    in_grammar = False
    if len(grammar.productions.get(cip.interface_hash, {})) > 1:
        in_grammar = True

    #sampler._samplelog('cip_select orig: scpre:%r inGramar:%r' % (score_ok, in_grammar), level=5)

    #return in_grammar and score_ok
    return in_grammar, score_ok

## SELECTING A CIP FROM THE GRAMMAR




def _select_cips( cip, decomposer, sampler):
    """

    Parameters
    ----------
    cip: CoreInterfacePair
        the cip we selected from the graph
    graphmancips: CIPs
        found in the grammar that can replace the input cip
    Returns
    -------
    yields CIPs
    """

    if not cip:
        raise Exception('select randomized cips from grammar got bad cip')

    # get core hashes
    core_hashes = sampler.lsgg.productions[cip.interface_hash].keys()
    if cip.core_hash in core_hashes:
        core_hashes.remove(cip.core_hash)

    # get values and yield accordingly
    values = _core_values(cip, core_hashes, decomposer.base_graph(), sampler)
    
    #print "I AM THE CHIPSELECTOR AND I HAVE %s %s" % (str(core_hashes),str(values))

    for core_hash in probabilistic_choice(values, core_hashes):
        # print values,'choose:', values[core_hashes.index(core_hash)]
        yield sampler.lsgg.productions[cip.interface_hash][core_hash]
        if sampler.quick_skip_orig_cip:
            yield StopIteration

def _core_values( cip, core_hashes, graph, sampler):
    '''
    assign probability values to each hash.
    elsewhere the new cip is picked based on these.

    Parameters
    ----------
    cip: cip
        that will be replaced
    core_hashes
        hashes of the available replacements
    graph
        the current graph

    Returns
    -------
        array  with probability value for each core_hash
    '''
    core_weights = []

    if sampler.probabilistic_core_choice:
        for core_hash in core_hashes:
            core_weights.append(sampler.lsgg.frequency[cip.interface_hash][core_hash])
    elif sampler.core_choice_bytrial:
        core_weights= map( lambda x: sampler.lsgg.productions[cip.interface_hash][x].bytrialscore*100 ,core_hashes)
        #print '*'*80
        #print 'cip_select'
        #print core_weights
    elif sampler.score_core_choice:
        for core_hash in core_hashes:
            core_weights.append(sampler.lsgg.score_core_dict[core_hash])

    elif sampler.size_constrained_core_choice > -1:
        unit = 100 / float(sampler.size_constrained_core_choice*2 + 1)
        goal_size = sampler.seed_size
        current_size = len(graph)

        for core in core_hashes:
            # print unit, self.lsgg.core_size[core] , cip.core_nodes_count , current_size , goal_size
            predicted_size = sampler.lsgg.core_size[core] - cip.core_nodes_count + current_size
            value = max(0, 100 - (abs(goal_size - predicted_size) * unit))
            core_weights.append(value)
    else:
        #print 'core weight is uniform'
        core_weights = [1] * len(core_hashes)

    if sampler.size_diff_core_filter > -1:
        # resultsizediff=  graphlen+new_core-oldcore-seed..
        # x is that without the new_core size:)
        x = len(graph) - sampler.seed_size - cip.core_nodes_count
        sizecheck = lambda core: abs(x + sampler.lsgg.core_size[core]) <= sampler.size_diff_core_filter
        #core_hashes = [core_hash for core_hash in core_hashes if sizecheck(core_hash)]
        for i,core in enumerate(core_hashes):
            if sizecheck(core)==False:
                core_weights[i]=0

    return core_weights

def probabilistic_choice(values, core_hashes):
    '''
    so you have a list of core_hashes
    now for every core_hash put a number in a rating list
    we will choose one according to the probability induced by those numbers


    Parameters
    ----------
    values: list with numbers for each cip
    core_hashes: list of core hashes

    Returns
    -------
        yields core hash according to propability induced by the values.

    '''
    ratings_sum = sum(values)
    # while there are cores
    while core_hashes and ratings_sum > 0.0:
        # get a random one by frequency
        rand = random.uniform(0.0, ratings_sum)
        if rand == 0.0:
            break
        current = 0.0
        i = -1
        while current < rand:
            current += values[i + 1]
            i += 1
        # yield and delete
        yield core_hashes[i]
        ratings_sum -= values[i]
        del values[i]
        del core_hashes[i]

