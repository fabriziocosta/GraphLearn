'''
functions that alter a graph and are copied from EDeN
'''

import numpy as np
from eden import fast_hash_2
from scipy.sparse import csr_matrix
import logging
logger = logging.getLogger(__name__)

'''
label preprocessing
'''
def _label_preprocessing(graph, label_size=1, key_label='label', key_entity='entity', discretizers={'entity':[]}, bitmask=2 ** 20 - 1):
    try:
        graph.graph['label_size'] = label_size
        for n, d in graph.nodes_iter(data=True):
            # for dense or sparse vectors
            if isinstance(d[key_label], list) or \
                    isinstance(d[key_label], dict):
                node_entity, data = _extract_entity_and_label(d, key_entity,key_label)
                if isinstance(d[key_label], list):
                    data = np.array(data, dtype=np.float64).reshape(1, -1)
                if isinstance(d[key_label], dict):
                    data = _convert_dict_to_sparse_vector(data)
                # create a list of integer codes of size: label_size
                # each integer code is determined as follows:
                # for each entity, use the correspondent
                # discretizers[node_entity] to extract
                # the id of the nearest cluster centroid, return the
                # centroid id as the integer code
                hlabel = []
                for i in range(label_size):
                    if len(discretizers[node_entity]) < i:
                        len_mod = \
                            len(discretizers[node_entity])
                        raise Exception('Error: discretizers for node entity: %s \
                            has length: %d but component %d was required'
                                        % (node_entity, len_mod, i))
                    predictions = \
                        discretizers[node_entity][i].predict(data)
                    if len(predictions) != 1:
                        raise Exception('Error: discretizer has not \
                            returned an individual prediction but\
                            %d predictions' % len(predictions))
                    discretization_code = predictions[0] + 1
                    code = fast_hash_2(hash(node_entity),
                                       discretization_code,
                                       bitmask)
                    hlabel.append(code)
                graph.node[n]['hlabel'] = hlabel
            elif isinstance(d[key_label], basestring):
                # copy a hashed version of the string for a number of times
                # equal to self.label_size in this way qualitative
                # ( i.e. string ) labels can be compared to the
                # discretized labels
                hlabel = int(hash(d[key_label]) & bitmask) + 1
                graph.node[n]['hlabel'] = [hlabel] * label_size
            else:
                raise Exception('ERROR: something went wrong, type of node label is unknown: \
                    %s' % d[key_label])
    except Exception as e:
        logger.debug('Failed iteration. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def _extract_entity_and_label(d, key_entity,key_label):
    # determine the entity attribute
    # if the vertex does not have a 'entity' attribute then provide a
    # default one
    if d.get(key_entity, False):
        node_entity = d[key_entity]
    else:
        if isinstance(d[key_label], list):
            node_entity = 'vector'
        elif isinstance(d[key_label], dict):
            node_entity = 'sparse_vector'
        else:
            node_entity = 'default'
    data = d[key_label]
    return node_entity, data


def _convert_dict_to_sparse_vector( feature_row, bitmask=2**20-1):
    feature_size= bitmask+2
    data, row, col = [], [], []
    if len(feature_row) == 0:
        # case of empty feature set for a specific instance
        row.append(0)
        col.append(0)
        data.append(0)
    else:
        for feature in feature_row:
            row.append(0)
            col.append(int(hash(feature) & bitmask) + 1)
            data.append(feature_row[feature])
    vec = csr_matrix((data, (row, col)), shape=(1, feature_size))
    return vec


