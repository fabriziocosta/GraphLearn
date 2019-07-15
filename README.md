[![DOI](https://zenodo.org/badge/33873956.svg)](https://zenodo.org/badge/latestdoi/33873956)

# GraphLearn
Learn how to construct graphs given representative examples.

Discriminative systems that can deal with graphs in input are known, however, generative or constructive approaches that can sample graphs from empirical distributions are less developed. This is a Metropolis–Hastings approach that uses a novel type of graph grammar to efficiently learn proposal distributions in a data driven fashion.


# References
Costa, Fabrizio. "Learning an efficient constructive sampler for graphs." Artificial Intelligence (2016). [link](http://www.sciencedirect.com/science/article/pii/S0004370216000138)



# Python3 

This demonstrates the grammar which is the heart of graphlearn. 
Sampling as in the py2 example is demonstrated in graphlearn/sample.py.

```python
from graphlearn.lsgg import lsgg
from graphlearn.util import util
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




# Results on MOSES benchmark

| Meassure     | aae     | char\_rnn | vae       | organ     | baseline   | coarsened  |
|--------------|---------|----------|-----------|-----------|------------|------------|
| valid        | 0.604   | 0.351    | 0.001     | 0.108     | 1.0        | 1.0        |
| unique@1000  | 0.358   | 1.0      | 1.0       | 0.944     | 1.0        | 1.0        |
| unique@10000 | 0.358   | 1.0      | 1.0       | 0.944     | 1.0        | 1.0        |
| FCD/Test     | 33.985  | 6.136    | nan       | 33.185    | 34.845     | 35.778     |
| SNN/Test     | 0.468   | 0.369    | 0.090     | 0.266     | 0.204      | 0.217      |
| Frag/Test    | 0.908   | 0.982    | 0.0       | 0.558     | 0.919      | 0.893      |
| Scaf/Test    | 0.0480  | 0.265    | nan       | 0.0       | 0.0        | 0.0        |
| FCD/TestSF   | 35.092  | 6.949    | nan       | 33.994    | 35.994     | 36.377     |
| SNN/TestSF   | 0.448   | 0.364    | 0.091     | 0.261     | 0.202      | 0.215      |
| Frag/TestSF  | 0.905   | 0.975    | 0.0       | 0.558     | 0.910      | 0.891      |
| Scaf/TestSF  | 0.0     | 0.022    | nan       | 0.0       | 0.0        | 0.0        |
| IntDiv       | 0.695   | 0.849    | 0.0       | 0.847     | 0.855      | 0.829      |
| IntDiv2      | 0.659   | 0.836    | 0.0       | 0.808     | 0.850      | 0.824      |
| Filters      | 0.957   | 0.934    | 1.0       | 0.583     | 0.079      | 0.872      |
| logP         | 0.0146  | 0.0839   | 8.752     | 0.388     | 1.186      | 3.260      |
| SA           | 0.311   | 0.081    | 0.266     | 0.438     | 11.714     | 9.176      |
| QED          | 0.002   | 0.0002   | 0.185     | 0.05145   | 0.315      | 0.279      |
| NP           | 0.306   | 0.099    | 5.090     | 2.971     | 4.025      | 3.325      |
| weight       | 323.785 | 52.685   | 76736.756 | 24619.107 | 66388.0129 | 140922.662 |







