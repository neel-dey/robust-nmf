# Robust-NMF
Python PyTorch (GPU) and/or NumPy (CPU)-based implementation of Févotte and Dobigeon's robust-NMF algorithm appearing in "Nonlinear hyperspectral unmixing with robust nonnegative matrix factorization." appearing in the IEEE Transactions on Image Processing, 2015. arXiv pre-print [here](https://arxiv.org/pdf/1401.5649.pdf).

Due to the simple multiplicative and mostly element-wise updates, the majority of operations are significantly faster on Graphics Processing Units (GPUs) and are thus implemented in PyTorch 1.0. If a GPU is not available, a NumPy backend is provided as well. 

The original MATLAB implementation is available at the authors websites [here](http://dobigeon.perso.enseeiht.fr/applications/app_hyper_rLMM.html) and [here](https://www.irit.fr/~Cedric.Fevotte/extras/tip2015/code.zip). 

## Features
1. Implements the low-rank and sparse non-negative matrix factorization proposed in the above paper. Through a suitable choice of the parameter beta, several assumptions on noise can be made. For example, beta = 2 corresponds to Gaussian noise, beta = 1 corresponds to Poisson noise and values in between interpolate between assumptions. Further, the choice of the L-2,1 norm enforces structured group sparsity on the outliers.

2. For a matrix of size (26x1447680), the PyTorch fp32 implementation is 58-64X faster on a GPU than the original MATLAB fp64 CPU implementation. At fp64, it is 32-35X faster than MATLAB on a desktop CPU. (more details below)

3. The code provides several initialization strategies, including:
    * **Random initializations**: drawn uniformly from [0,1).
    * **NMF**: Euclidean NMF as in [Cichocki and Phan, 2009.](http://www.bsp.brain.riken.jp/publications/2009/Cichocki-Phan-IEICE_col.pdf)
    * **beta-NMF**: Beta NMF as in [Févotte and Idier, 2011.](https://arxiv.org/pdf/1010.1763.pdf)
    * **nndsvd(ar)**: modified NNDSVD as in [Boutsidis and Gallopoulos, 2008.](http://www.boutsidis.org/Boutsidis_PRE_08.pdf) The modified version is used as the original version would cause saturation with zeros in a multiplicative setting.
    * **User provided** initializations.

4. Makes the simplex (non-negative and sum-to-one) constraint on coefficients optional.

## Installation and dependencies
The easiest method of installing the relevant dependencies is if you use Anaconda with the provided environment.yml file. If so, simply clone or download this repository, navigate to the folder, open a terminal and type:
```bash
conda env create -f environment.yml
```
That will create a conda virtual environment with all required dependencies. Otherwise, the major dependencies are:
  * pytorch 1.0
  * numpy
  * scikit-learn

If you're just using the NumPy version, you only need the last two. NOTE: This was developed in PyTorch 1.0 but should be fine with 0.4 as well.

## Interface
To switch between NumPy and PyTorch for computations is just a simple change in importing.

For NumPy:
```python
from backends.numpy_functions import *
```

For PyTorch:
```python
from backends.torch_functions import *
```

A couple of minimal working examples with random data are provided in the `./example_notebooks/` folder.

## Speed comparison details
The MATLAB fp64 implementation was tested on an Intel i7-5930K overclocked to 4.4 GHz. The PyTorch fp64 and fp32 implementations were performed on a stock NVIDIA Tesla P100. The matrix was of size (26,1447680) and the parameters for the algorithm were a rank of 3, beta = 1.5, maximum iterations = 100 and lambda = 1.

  * *Without a simplex constraint*:
  At fp64, MATLAB: 398.71 seconds; PyTorch: 11.25 seconds. At fp32, PyTorch brings this down to 6.19 seconds.
  * *With a simplex constraint*:
  At fp64, MATLAB: 414.41 seconds; PyTorch: 12.77 seconds. At fp32, PyTorch brings this down to 7.11 seconds.

The Tesla P100 is specifically meant for scientific workloads, so fp64 performance is only half of fp32 in terms of TFLOPs. However, appreciable speedups can be achieved even with an entry-level gaming GPU from two generations ago with low fp64 performance. With an NVIDIA GTX 970, at fp32 there is a 29-34X speedup over MATLAB on the CPU (depending on whether you use the simplex constraint or not). At fp64, it's less significant with a 4-4.7X speedup over the CPU.

NOTE 1: since R2018a (or R2017b, not sure) which was used here, MATLAB does block processing on element-wise operations automatically, leading to significant improvements in speed for rNMF.

NOTE 2: MATLAB can use GPUs as well via the ```gpuArray()`` data type, but only at fp32, which may be suboptimal for some optimization problems.

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
