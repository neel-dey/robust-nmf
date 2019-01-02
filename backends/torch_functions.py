#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python/PyTorch port of [1]. Original MATLAB code available at the authors
websites. This code implements the generalized beta divergence, as in the
authors technical report [2].

Algorithms available for the initializations of rNMF are listed in [3-5].

Created on Sat Dec 29 2018.

If you find bugs and/or limitations, please email neel DOT dey AT nyu DOT edu.

REFERENCES:
    [1] Févotte, Cédric, and Nicolas Dobigeon. "Nonlinear hyperspectral
    unmixing with robust nonnegative matrix factorization." IEEE Transactions
    on Image Processing 24.12 (2015): 4810-4819.
    [2] Févotte, Cédric, and Nicolas Dobigeon. "Nonlinear hyperspectral
    unmixing with robust nonnegative matrix factorization." arXiv preprint
    arXiv:1401.5649 (2014).
    [3] Cichocki, Andrzej, and Anh-Huy Phan. "Fast local algorithms for large
    scale nonnegative matrix and tensor factorizations." IEICE transactions on
    fundamentals of electronics, communications and computer sciences 92.3
    (2009): 708-721.
    [4] Févotte, Cédric, and Jérôme Idier. "Algorithms for nonnegative matrix
    factorization with the β-divergence." Neural computation 23.9 (2011):
    2421-2456.
    [5] Boutsidis, Christos, and Efstratios Gallopoulos. "SVD based
    initialization: A head start for nonnegative matrix factorization." Pattern
    Recognition 41.4 (2008): 1350-1362.
"""

import torch
from torch.nn.functional import normalize
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition.nmf import _initialize_nmf


def robust_nmf(data, rank, beta, init, reg_val, sum_to_one, tol, max_iter=1000,
               print_every=10, user_prov=None):
    '''
    This function performs the robust NMF algorithm.

    Input:
        1. data: data to be factorized. WIP: based on the data type of 'data',
        all computations performed at fp32 or fp64. fp64 implemented currently.
        2. rank: rank of the factorization/number of components.
        3. beta: parameter of the beta-divergence used.
            Special cases:
            beta = 2: Squared Euclidean distance (Gaussian noise assumption)
            beta = 1: Kullback-Leibler divergence (Poisson noise assumption)
            beta = 0: Itakura-Saito divergence (multiplicative gamma noise
            assumption)
        4. init: Initialization method used for robust NMF.
            init == 'random': Draw uniform random values (recommended).
            init == 'NMF': Uses a small run of regular NMF to get initial
            values and initializes outliers uniformly at random.
            init == 'bNMF': Uses a small run of beta NMF to get initial values
            and initializes outliers uniformly at random.
            init == 'nndsvdar': Uses Boutsidis' modified algorithm and
            initializes outliers uniformly at random.
            init == 'user': the user can provide their own initialization in
            the form of a python dictionary with the keys: 'basis', 'coeff' and
            'outlier'.
        5. reg_val: Weight of L-2,1 regularization.
        6. sum_to_one: flag indicating whether a sum-to-one constraint is to be
        applied on the factor matrices.
        7. tol: tolerance on the iterative optimization. Recommended: 1e-7.
        8. max_iter: maximum number of iterations.
        9. print_every: Number of iterations at which to show optimization
        progress.

    Output:
        1. basis: basis matrix of the factorization.
        2. coeff: coefficient matrix of the factorization.
        3. outlier: sparse outlier matrix.
        4. obj: objective function progress.

    NOTE: init == 'bNMF' applies the same beta parameter as required for rNMF,
    which is nice, but is slow due to multiplicative updates
    '''

    # Utilities:
    # Defining epsilon to protect against division by zero:
    if data.type() == 'torch.cuda.FloatTensor':
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Initialize rNMF:
    basis, coeff, outlier = initialize_rnmf(data, rank, init, beta,
                                            sum_to_one, user_prov)

    # Set up for the algorithm:
    # Initial approximation of the reconstruction:
    data_approx = basis@coeff + outlier + eps
    fit = torch.zeros(max_iter+1)
    obj = torch.zeros(max_iter+1)

    # Monitoring convergence:
    fit[0] = beta_divergence(data, data_approx, beta)
    obj[0] = (fit[0] +
              reg_val*torch.sum(torch.sqrt(torch.sum(outlier**2, dim=0))))

    # Print initial iteration:
    print('Iter = 0; Obj = {}'.format(obj[0]))

    for iter in range(max_iter):
        # Update the outlier matrix:
        outlier = update_outlier(data, data_approx, outlier, beta, reg_val)
        data_approx = basis@coeff + outlier + eps  # Update reconstuction

        # Update the coefficient matrix:
        coeff = update_coeff(data, data_approx, beta, basis, coeff, sum_to_one)
        data_approx = basis@coeff + outlier + eps  # Update reconstruction

        # Update the basis matrix:
        basis = update_basis(data, data_approx, beta, basis, coeff)
        data_approx = basis@coeff + outlier + eps  # Update reconstruction

        # Monitor optimization:
        fit[iter+1] = beta_divergence(data, data_approx, beta)
        obj[iter+1] = (fit[iter+1] +
                       reg_val*torch.sum(torch.sqrt(torch.sum(outlier**2,
                                                              dim=0))))

        if iter % print_every == 0:  # print progress
            print('Iter = {}; Obj = {}; Err = {}'.format(iter+1, obj[iter+1],
                  torch.abs((obj[iter]-obj[iter+1])/obj[iter])))

        # Termination criterion:
        if torch.abs((obj[iter]-obj[iter+1])/obj[iter]) <= tol:
            print('Algorithm converged as per defined tolerance')
            break

        if iter == (max_iter - 1):
            print('Maximum number of iterations acheived')

    # In case the algorithm terminated early:
    obj = obj[:iter]
    fit = fit[:iter]

    return basis, coeff, outlier, obj


def initialize_rnmf(data, rank, alg, beta=2, sum_to_one=0, user_prov=None):
    '''
    This function retrieves factor matrices to initialize rNMF. It can do this
    via the following algorithms:
        1. 'random': draw uniform random values.
        2. 'NMF': initialize with 200 iterations of regular NMF.
        3. 'bNMF': initialize with 200 iterations of beta NMF.
        4. 'nndsvdar': initialize with Boutsidis' modified algorithm. (classic
        nndsvd will cause issues with division by zero)
        5. 'user': provide own initializations. Must be passed in 'user_prov'
        as a dictionary with the format:
            user_prov['basis'], user_prov['coeff'], user_prov['outlier']

    'NMF', 'bNMF', 'nndsvdar' cause a switch to NumPy as these algorithms do
    not have PyTorch implementations, before going back to PyTorch. This
    shouldn't be a problem as sklearn's implementations are quite efficient and
    these are just initializations.

    Input:
        1. data: data to be factorized.
        2. rank: rank of the factorization/number of components.
        3. alg: Algorithm to initialize factorization. Either 'random', 'NMF',
        or 'bNMF'. 'bNMF' is the slowest option.
        4. beta: parameter for beta-NMF. Ignored if not provided.
        5. sum_to_one: binary flag indicating whether a simplex constraint will
        be later applied on the coefficient matrix.
        6. user_prov: if alg == 'user', then this is the dictionary containing
        the user provided initial values to use. Mandatory keys: 'basis',
        'coeff', and 'outlier'.

    Output:
        1. basis: initial basis matrix.
        2. coeff: initial coefficient matrix.
        3. outlier: initial outlier matrix.

    This can use a small run of regular/beta NMF to initialize rNMF via 'alg'.
    If a longer run is desired, or other parameters of sklearn's NMF are
    desired, modify the code below in the else block. NMF itself is very
    initialization sensitive. Here, we use Boutsidis, et al.'s NNDSVD algorithm
    to initialize it.

    Empirically, random initializations work well for rNMF.

    This initializes the outlier matrix as uniform random values.
    '''

    # Utilities:
    # Defining epsilon to protect against division by zero:
    if data.type() == 'torch.cuda.FloatTensor':
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Initialize outliers with uniform random values:
    outlier = torch.rand(data.size()[0], data.size()[1])

    # Initialize basis and coefficients:
    if alg == 'random':
        print('Initializing rNMF uniformly at random.')
        basis = torch.rand(data.size()[0], rank)
        coeff = torch.rand(rank, data.size()[1])

        # Rescale coefficients if they will have a simplex constraint later:
        if sum_to_one == 1:
            coeff = normalize(coeff, p=1, dim=0, eps=eps)

        return basis+eps, coeff+eps, outlier+eps

    elif alg == 'bNMF':
        # Switching to CPU as there is no GPU implementation of bNMF:

        # NNDSVDar used to initialize beta-NMF as multiplicative algorithms do
        # not like zero values and regular NNDSVD causes sparsity.
        print('Initializing rNMF with beta-NMF. Switching to NumPy.')
        model = NMF(n_components=rank, init='nndsvdar', beta_loss=beta,
                    solver='mu', verbose=True)
        basis = model.fit_transform(data.cpu().numpy())
        coeff = model.components_

        # Bringing output back into the GPU:
        print('Done. Switching back to PyTorch.')
        if data.type() == 'torch.cuda.FloatTensor':
            basis = torch.tensor(basis, dtype=torch.float32).cuda()
            coeff = torch.tensor(coeff, dtype=torch.float32).cuda()
        else:
            basis = torch.from_numpy(basis).cuda()
            coeff = torch.from_numpy(coeff).cuda()

        # Rescale coefficients if they will have a simplex constraint later:
        if sum_to_one == 1:
            coeff = normalize(coeff, p=1, dim=0, eps=eps)

        return basis+eps, coeff+eps, outlier+eps

    elif alg == 'NMF':
        # Switching to CPU as there is no GPU implementation of NMF:

        print('Initializing rNMF with NMF. Switching to NumPy.')
        model = NMF(n_components=rank, init='nndsvd', verbose=True)
        basis = model.fit_transform(data.cpu().numpy())
        coeff = model.components_

        # Bringing output back into the GPU:
        print('Done. Switching back to PyTorch.')
        if data.type() == 'torch.cuda.FloatTensor':
            basis = torch.tensor(basis, dtype=torch.float32).cuda()
            coeff = torch.tensor(coeff, dtype=torch.float32).cuda()
        else:
            basis = torch.from_numpy(basis).cuda()
            coeff = torch.from_numpy(coeff).cuda()

        # Rescale coefficients if they will have a simplex constraint later:
        if sum_to_one == 1:
            coeff = normalize(coeff, p=1, dim=0, eps=eps)

        return basis+eps, coeff+eps, outlier+eps

    elif alg == 'nndsvdar':
        # Switching to CPU as there is no GPU implementation of nndsvdar:
        print('Initializing rNMF with nndsvdar. Switching to NumPy.')
        basis, coeff = _initialize_nmf(data.cpu().numpy(), n_components=rank)

        # Bringing output back into the GPU:
        print('Done. Switching back to PyTorch.')
        if data.type() == 'torch.cuda.FloatTensor':
            basis = torch.tensor(basis, dtype=torch.float32).cuda()
            coeff = torch.tensor(coeff, dtype=torch.float32).cuda()
        else:
            basis = torch.from_numpy(basis).cuda()
            coeff = torch.from_numpy(coeff).cuda()

        # Rescale coefficients if they will have a simplex constraint later:
        if sum_to_one == 1:
            coeff = normalize(coeff, p=1, dim=0, eps=eps)

        return basis+eps, coeff+eps, outlier+eps

    elif alg == 'user':
        print('Initializing rNMF with user provided values.')

        # Make sure that the initialization provided is in the correct format:
        if user_prov is None:
            raise ValueError('You forgot the dictionary with the data')

        elif type(user_prov) is not dict:
            raise ValueError('Initializations must be in a dictionary')

        elif ('basis' not in user_prov or 'coeff' not in user_prov or
              'outlier' not in user_prov):
            raise ValueError('Wrong format for initialization dictionary')

        elif (user_prov['basis'].type() != data.type() or
              user_prov['coeff'].type() != data.type() or
              user_prov['outlier'].type() != data.type()):
            raise ValueError('Initializations must the same dtype as data')

        return user_prov['basis'], user_prov['coeff'], user_prov['outlier']

    else:
        # Making sure the user doesn't do something unexpected:
        # Inspired by how sklearn deals with this:
        raise ValueError(
            'Invalid algorithm (typo?): got %r instead of one of %r' %
            (alg, ('random', 'NMF', 'bNMF', 'nndsvdar', 'user')))


def beta_divergence(mat1, mat2, beta):
    '''
    This follows the definition of the beta divergence used by Fevotte, et al.
    Another definition of the beta divergence used by Amari, et al. shifts the
    values of beta by one.

    Input:
        1. mat1, mat2: matrices between which to calculate the beta divergence
        2. beta: parameter of the beta divergence

    Output:
        1. beta_div: the beta-divergence between mat1 and mat2.

    Special cases of beta:
        1. beta = 2 : Squared Euclidean Distance (Gaussian noise assumption)
        2. beta = 1 : Kullback-Leibler Divergence (Poisson noise assumption)
        3. beta = 0 : Itakura-Saito Divergence (multiplicative gamma noise
        assumption)

    NOTE: If beta = 0, the data cannot contain any zero values. If beta = 1,
    Fevotte and Dobigeon explicitly work around zero values in their version of
    the KL-divergence as shown below. beta = 2 is just the squared Frobenius
    norm of the difference between the two matrices. With the squaring, it is
    no longer an actual distance metric.

    Beta values in between the above interpolate between assumptions.
    '''

    # Utilities:
    # Defining epsilon to protect against division by zero:
    if mat1.type() == 'torch.cuda.FloatTensor':
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Inline function for vectorizing arrays for readability:
    vec = lambda X: X.flatten()

    # Main section:
    # If/else through the special limiting cases of beta, otherwise use the
    # last option:

    if beta == 2:
        # Gaussian assumption.
        beta_div = 0.5*(torch.norm(mat1 - mat2, p='fro')**2)

    elif beta == 1:
        # Poisson assumption.

        # Finding elements that would cause a division by zero/issues with log:
        zeromask = mat1 <= eps
        onemask = ~zeromask

        beta_div = (torch.sum((mat1[onemask] *
                              torch.log(mat1[onemask]/mat2[onemask])) -
                              mat1[onemask] + mat2[onemask]) +
                    torch.sum(mat2[zeromask]))

    elif beta == 0:
        # Multiplicative gamma assumption.
        beta_div = torch.sum(vec(mat1)/vec(mat2) -
                             torch.log(vec(mat1)/vec(mat2))) - len(vec(mat1))

    else:
        # General case.
        beta_div = torch.sum(vec(mat1)**beta + (beta-1)*vec(mat2)**beta
                             - beta*vec(mat1)*(vec(mat2))**(beta-1))\
                          / (beta*(beta-1))

    return beta_div


def update_basis(data, data_approx, beta, basis, coeff):
    '''
    This function updates the basis vectors of the approximation.
    In the paper, this is the M matrix.

    Input:
        1. data: data matrix to be factorized.
        2. data_approx: current approximation of the model to the data.
        3. beta: parameter of the beta-divergence.
        4. basis: current estimate of the basis matrix.
        5. coeff: current estimate of the coefficent matrix.

    Output:
        Multiplicative update for basis matrix.
    '''
    return basis * ((data*(data_approx**(beta-2))@coeff.t()) /
                    ((data_approx**(beta-1))@coeff.t()))


def update_coeff(data, data_approx, beta, basis, coeff, sum_to_one):
    '''
    This function updates the coefficient matrix of the approximation.
    In the paper, this is the A matrix.

    Input:
        1. data: data matrix to be factorized.
        2. data_approx: current approximation of the model to the data.
        3. beta: parameter of the beta-divergence.
        4. basis: current estimate of the basis matrix.
        5. coeff: current estimate of the coefficent matrix.
        6. sum_to_one: binary flag indicating whether a simplex constraint is
        applied on the coefficents.

    Output:
        Multiplicative update for coefficient matrix.
    '''

    # Using inline functions for readability:
    bet1 = lambda X: X**(beta-1)
    bet2 = lambda X: X**(beta-2)

    if sum_to_one == 1:

        Gn = ((basis.t())@(data*bet2(data_approx)) +
              torch.sum((basis@coeff)*bet1(data_approx), dim=0))
        Gp = ((basis.t())@bet1(data_approx) +
              torch.sum((basis@coeff)*data*bet2(data_approx), dim=0))
        coeff = coeff*(Gn/Gp)

        return normalize(coeff, p=1, dim=0)

    elif sum_to_one == 0:
        return coeff * (((basis.t())@(data*bet2(data_approx))) /
                        ((basis.t())@bet1(data_approx)))


def update_outlier(data, data_approx, outlier, beta, reg_val):
    '''
    This function updates the outlier matrix within the approximation.
    In the paper, this is the R matrix.

    Input:
        1. data: data matrix to be factorized.
        2. data_approx: current approximation of the model to the data.
        3. outlier: current estimate of the outlier matrix.
        4. beta: parameter of the beta-divergence.
        5. reg_val: strength of L-2,1 regularization on outliers.

    Output:
        Multiplicative update for outlier matrix.
    '''
    # Utilities:
    # Defining epsilon to protect against division by zero:
    if data.type() == 'torch.cuda.FloatTensor':
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Using inline functions for readability:
    bet1 = lambda X: X**(beta-1)
    bet2 = lambda X: X**(beta-2)

    return outlier * ((data*bet2(data_approx)) / (bet1(data_approx) +
                                                  reg_val*normalize(outlier,
                                                                    p=2,
                                                                    dim=0,
                                                                    eps=eps)))
