[![DOI](https://zenodo.org/badge/33873956.svg)](https://zenodo.org/badge/latestdoi/33873956)

# GraphLearn
Learn how to construct graphs given representative examples.

Discriminative systems that can deal with graphs in input are known, however, generative or constructive approaches that can sample graphs from empirical distributions are less developed. This is a Metropolis–Hastings approach that uses a novel type of graph grammar to efficiently learn proposal distributions in a data driven fashion.


# References
Costa, Fabrizio. "Learning an efficient constructive sampler for graphs." Artificial Intelligence (2016). [link](http://www.sciencedirect.com/science/article/pii/S0004370216000138)



# Python3 

This demonstrates the grammar which is the heart of graphlearn. 
Sampling as in the py2 example is demonstrated in graphlearn3/sample.py.

```python
from graphlearn3.lsgg import lsgg
from graphlearn3.util import util
import structout as so

# get graphs
gr = util.get_cyclegraphs()

#  induce a grammar, pick a graph, and apply all possible substitutions
mylsgg = lsgg()
mylsgg.fit(gr)
graphs =  list(mylsgg.neighbors(gr[0]))
so.gprint(graphs)
```

![''](https://raw.githubusercontent.com/fabriziocosta/GraphLearn/master/example.png)





# Python2 

```python

# set up graph source

from eden.io.gspan import gspan_to_eden
from itertools import islice
def get_graphs(dataset_fname='../../toolsdata/bursi.pos.gspan', size=100):
    return  islice(gspan_to_eden(dataset_fname),size)
    
# sample some graphs

from graphlearn.graphlearn import  Sampler
sampler=Sampler(n_steps=50)
samples = sampler.fit_transform(get_graphs())

# draw result

from graphlearn.utils import draw
for i in range(5):
        draw.graphlearn(samples.next())
```
![''](https://raw.githubusercontent.com/fabriziocosta/GraphLearn/master/example_py2.png)


## Install (Py2)

We only maintain the python3 version at this point. an outdated but detailed installation guide for the 
python2 version is available [here](install.md).


## Examples (py2)
See [here](https://github.com/fabriziocosta/GraphLearn_examples) for more examples. Examples still use python2... 

* [Introduction to GraphLearn](https://github.com/fabriziocosta/GraphLearn_examples/blob/master/notebooks/Introduction.ipynb)

* [CoreMorph -- enhanced grammar](https://github.com/fabriziocosta/GraphLearn_examples/blob/master/notebooks/CoreMorph.ipynb)

* [Interactive -- sample graphs step by step](https://github.com/fabriziocosta/GraphLearn_examples/blob/master/notebooks/simple_toys/interactive_creation.ipynb)

* [MultiGoal -- optimize towards multiple goals](https://github.com/fabriziocosta/GraphLearn_examples/blob/master/notebooks/SamplerCombiner.ipynb)

* [Cascading Learned Abstractions -- add learned layers to graphs](https://github.com/smautner/GraphLearn_examples/blob/master/notebooks/cascade.ipynb)













