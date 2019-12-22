# Speed up Cosine Similarity computations with Numba


As a Machine Learning Engineer, I work a good amount of time on scaling the ML Algorithms. And on one fine day, I got introduced to Numba. This repository contains the jupyter-notebook and the results of my experiment with [Numba](https://numba.pydata.org/).

To get a better overview, I would recommend reading this [article](dummy).

----------
### Experiment

To verify the difference in time taken to compute cosine similarity between two numpy arrays with and without Numba.

----------
### Code

``` python
def cosine_similarity(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta
```

To use Numba, install Numba using pip.
``` python
pip3 install numba
```
Now from Numba, import the Just In Time compiler and add the decorator on top of the cosine similarity function.
``` python
from numba import jit

@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta
```


----------
### Results


|No. of computations|Without Numba | With Numba
|--|--|--|
|1|83.6 µs|1.11 µs|
|100|8.34 ms|70.9 µs|
|1000|86.6 ms|706 µs|
|10000|830 ms|7.09 ms|
|100000|8.08 s|70.2 ms|
|1000000|1min 27s|699 ms|

The vectors we used for the above computations are of 50 dimensions.

----------
### Conclusion

There is a significant decrease in computation time when we compile the python function with Numba.

In this experiment, we just scratched the surface. There is lot more you can do with numba using other decorators and the parameters. I will leave you all to explore that by yourselves.

----------
### Author

[Pranay Chandekar](https://in.linkedin.com/in/pranaychandekar)




