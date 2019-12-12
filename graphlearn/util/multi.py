

import multiprocessing as mp
def mpmap(func, iterable, chunksize=10, poolsize=2):
    """pmap."""
    pool = mp.Pool(poolsize)
    result = pool.map(func, iterable, chunksize=chunksize)
    pool.close()
    pool.join()
    return list(result)

