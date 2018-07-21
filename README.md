[![DOI](https://zenodo.org/badge/33873956.svg)](https://zenodo.org/badge/latestdoi/33873956)

# GraphLearn
Learn how to construct graphs given representative examples.

Discriminative systems that can deal with graphs in input are known, however, generative or constructive approaches that can sample graphs from empirical distributions are less developed. This is a Metropolis–Hastings approach that uses a novel type of graph grammar to efficiently learn proposal distributions in a data driven fashion.


# References
Costa, Fabrizio. "Learning an efficient constructive sampler for graphs." Artificial Intelligence (2016). [link](http://www.sciencedirect.com/science/article/pii/S0004370216000138)

# Usage

<!--
You want to install EDeN first:
```python
pip install git+https://github.com/fabriziocosta/EDeN.git --user
```
Then GraphLearn
```python
pip install git+https://github.com/fabriziocosta/GraphLearn.git --user
```
-->

#### Setting up a networkx graph generator

```python
%matplotlib inline
from eden.io.gspan import gspan_to_eden
from itertools import islice
def get_graphs(dataset_fname='../../toolsdata/bursi.pos.gspan', size=100):
    return  islice(gspan_to_eden(dataset_fname),size)
```

#### Sampling new graphs

Sampling is straight forward.
There are many options for the sampling process available.

```python
from graphlearn.graphlearn import  Sampler
sampler=Sampler(n_steps=50)
samples = sampler.fit_transform(get_graphs())

```

#### Drawing the result

Each sample output is a list of networkx graphs.
graphlearns draw function can print these more or less nicely.
```python
from graphlearn.utils import draw
for i in range(5):
        draw.graphlearn(samples.next())
```

![''](https://raw.githubusercontent.com/fabriziocosta/GraphLearn/master/example.png)

# Examples
See [here](https://github.com/fabriziocosta/GraphLearn_examples) for more examples.

* [Introduction to GraphLearn](https://github.com/fabriziocosta/GraphLearn_examples/blob/master/notebooks/Introduction.ipynb)

* [CoreMorph -- enhanced grammar](https://github.com/fabriziocosta/GraphLearn_examples/blob/master/notebooks/CoreMorph.ipynb)

* [Interactive -- sample graphs step by step](https://github.com/fabriziocosta/GraphLearn_examples/blob/master/notebooks/simple_toys/interactive_creation.ipynb)

* [MultiGoal -- optimize towards multiple goals](https://github.com/fabriziocosta/GraphLearn_examples/blob/master/notebooks/SamplerCombiner.ipynb)

* [Cascading Learned Abstractions -- add learned layers to graphs](https://github.com/smautner/GraphLearn_examples/blob/master/notebooks/cascade.ipynb)




# Install 

a complete install guide can be found [here](install.md).











