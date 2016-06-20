# GraphLearn
Learn how to construct graphs given representative examples



* [Introduction to GraphLearn](https://github.com/smautner/GraphLearn_examples/blob/master/notebooks/Introduction.ipynb)

* [CoreMorph -- enhanced grammar](https://github.com/smautner/GraphLearn_examples/blob/master/notebooks/CoreMorph.ipynb)

* [Interactive -- sample graphs step by step](https://github.com/smautner/GraphLearn_examples/blob/master/notebooks/simple_toys/interactive_creation.ipynb)

* [MultiGoal -- optimize towards multiple goals](https://github.com/smautner/GraphLearn_examples/blob/master/notebooks/SamplerCombiner.ipynb)





## This is a short introduction on how to use Graphlearn.


You want to install EDeN first:

```python
pip install git+https://github.com/fabriziocosta/EDeN.git --user
```


#### Setting up a networkx graph generator

```python
%matplotlib inline
from eden.converter.graph.gspan import gspan_to_eden
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
