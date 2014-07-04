# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:57:39 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
from scipy import ndimage
from ..utils import Dot, ObjImage, spatial_smoothing, corr_to_coef


def load(n_samples=100, shape=(30, 30, 1),
                           r2=.75,
                           sigma_spatial_smoothing=1,
                           object_pixel_ratio=.5,
                           objects=None,
                           random_seed=None):
    """Generates regression samples (images + target variable) and beta.

    Input features (X) covariance structure is controled both at a pixel
    level (spatial smoothing) and object level. Objects are component
    of pixels sharing a covariance that stem from a latent variable.
    beta is non null within objects (default is five dots).
    Then y is obtained with y = X * beta + noise, where beta is scalled such
    that r_square(y, X * beta) = r2 and noise is sampled according to N(0, 1).

    Parameters
    ----------
    n_samples: Integer. Number of samples. Default is 100.

    # TODO: This is wrong. Shape should be Z, Y, X.
    shape: Tuple or list with three integers. Order x, y, z shape each samples
            Default is (30, 30, 1).

    r2: Float. The desire R-squared (explained variance) ie.:
            r_square(y, X * beta) = r2 (Default is .75)

    sigma_spatial_smoothing: Float. Standard deviation for Gaussian kernel
            (default is 1). High value promotes spatial correlation pixels.

    object_pixel_ratio: Float. Controls the ratio between object-level signal
            and pixel-level signal for pixels within objects. If
            object_pixel_ratio == 1 then 100% of the signal of pixels within
            the same object is shared (ie.: no pixel level) signal. If
            object_pixel_ratio == 0 then all the signal is pixel specific.
            High object_pixel_ratio promotes spatial correlation between
            pixels of the same object.

    objects: List of objects. Objects carying information to be drawn in the
            image. If not provide a dice with five points (object) will be
            drawn. Point 1, 3, 4 are carying predictive information while
            point 2 is a suppressor of point 1 and point 5 is a suppressor of
            point 3. Object should implement "get_mask()" method, a have
            "is_suppressor" (bool) and "r" (ref to suppressor object,
            possibely None) attributes.

    random_seed: None or integer. See numpy.random.seed(). If not None, it can
            be used to obtain reproducable samples.

    Returns
    -------
    X3d: Numpy array of shape [n_sample, shape]. The input features.

    y: Numpy array of shape [n_sample, 1]. The target variable.

    beta3d: Numpy array of shape [shape,]. It is the beta such that
            y = X * beta + noise.

    Details
    -------
    The general procedure is:
        1) For each pixel i, Generate independant variables Xi ~ N(0, 1)
        2) Add object level structure corresponding to the five dots:
           - Sample five latent variables ~ N(0, 1): l1, l12, l3, l4, l45.
           - Pixel i of dots X1, X2, X3, X4, X5 are sampled as:
             X1i = l1 + l12 + Xi
             X2i = l12 + Xi
             X3i = 2 * l3 + Xi
             X4i = l4 + l45 + Xi
             X5i = l45 + Xi
             Note that:
             Pixels of dot X1 share a common variance that stem from l1 + l12.
             Pixels of dot X2 share a common variance that stem from l12.
             Pixels of dot X1 and pixel of dot X2 share a common variance that
             stem from l12.
             etc.
        4) Spatial Smoothing.
        5) Model: y = X beta + noise
        - Betas are null outside dots, and 1 or -1 depending on the dot:
            X1: 1, X2: -1, X3: 1, X4: 1, X5: -1.
        - Sample noise ~ N(0, 1)
        - Compute X beta then scale beta such that: r_squared(y, X beta) = r2
        Return X, y, beta

        Note that negative coeficients for X2 and X5 combined with the shared
        latent variable l12 and l45 make X2 and X5 suppressor regions ie.:
        regions are not correlated with y but
            y = X1 - X2 + X3 + X4 -X5 + noise
            y = l1 + l3 + l4 + noise
            So pixels of X2 and X5 are not correlated with the target y so they
            will not be detected by univariate analysis. However, they
            are usefull since they are suppressing unwilling variance that stem
            from latents l12 and l45.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plot
    >>> from  parsimony import datasets
    >>> n_samples = 100
    >>> shape = (11, 11, 1)
    >>> X3d, y, beta3d = datasets.regression.dice5.load(n_samples=n_samples,
    ...     shape=shape, r2=.5, random_seed=1)
    """
    sigma_pix = 1  # items std-dev
    mu_e = 0
    if shape[0] < 5 or shape[1] < 5:
        raise ValueError("Shape too small. The minimun is (5, 5, 0)")

    if len(shape) == 2:
        shape = tuple(list(shape) + [1])

    n_features = np.prod(shape)
    nx, ny, nz = shape

    ##########################################################################
    ## 1. Build images with signal => e_ij
    # Sample signal: X
    if random_seed is not None:  # If random seed, save current random state
        rnd_state = np.random.get_state()
        np.random.seed(random_seed)
    X = np.random.normal(mu_e, sigma_pix, n_samples * n_features)
    X3d = X.reshape(n_samples, nx, ny, nz)
    #########################################################################
    ## 2. Build Objects
    if objects is None:
        objects = dice_five_with_union_of_pairs(shape, beta=1.,
                                                std=sigma_pix)
    #########################################################################
    ## 3. Object-level structured signal
    X3d, support = ObjImage.object_model(objects, X3d)
    #########################################################################
    ## 4. Pixel-level signal structure: spatial smoothing
    if sigma_spatial_smoothing != 0:
        X3d = spatial_smoothing(X3d, sigma_spatial_smoothing, mu_e,
                                  sigma_pix)
    X = X3d.reshape((X3d.shape[0], np.prod(X3d.shape[1:])))
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    #########################################################################
    ## 5. Model: y = X beta + noise
    beta3d = np.zeros(X3d.shape[1:])

    for k in xrange(len(objects)):
        o = objects[k]
        beta3d[o.get_mask()] += o.beta
    beta3d = ndimage.gaussian_filter(beta3d, sigma=sigma_spatial_smoothing)
    beta = beta3d.ravel()
    # Fix a scaling to get the desire r2, ie.:
    # y = coef * X * beta + noise
    # Fix coef such r2(y, coef * X * beta) = r2
    X = X3d.reshape(n_samples, np.prod(shape))
    Xbeta = np.dot(X, beta)

    if r2 < 1:
        noise = np.random.normal(0, 1, Xbeta.shape[0])
        coef = corr_to_coef(v_x=np.var(Xbeta), v_e=np.var(noise),
                     cov_xe=np.cov(Xbeta, noise)[0, 1], cor=np.sqrt(r2))
        beta *= coef
        y = np.dot(X, beta) + noise
    else:
        y = np.dot(X, beta)

    if False:
        import pylab as plt
        X = X3d.reshape((n_samples, nx * ny))
        Xc = (X - X.mean(axis=0)) / X.std(axis=0)
        yc = (y - y.mean()) / y.std()
        cor = np.dot(Xc.T, yc).reshape(nx, ny) / y.shape[0]
        cax = plt.matshow(cor, cmap=plt.cm.coolwarm)
        plt.colorbar(cax)
        plt.show()

    if random_seed is not None:   # If random seed, restore random state
        np.random.set_state(rnd_state)

    return X3d, y.reshape((n_samples, 1)), beta3d


############################################################################
## Objects builder
def dice_five_with_union_of_pairs(shape, std, beta):
    """Seven objects, five dot + union1 = dots 1 + 2 and union2 = dots 4 + 5
    1, 3 and 4 have beta == beta
    2, 5  have beta = -beta/2
    union1 and union2 have beta = 0

    Examples
    --------
    #shape = (5, 5, 1)
    beta = 1
    std = 1
    noise = np.zeros(shape)
    info = np.zeros(shape)
    for o in dice_five_with_union_of_pairs(shape, beta, std):
       noise[o.get_mask()] += o.std
       info[o.get_mask()] += o.beta
       print o.std, o.beta
    plot = plt.subplot(121)
    import matplotlib.pyplot as plt
    cax = plot.matshow(noise.squeeze())
    plt.colorbar(cax)
    plt.title("Noise sum coeficients")
    plot = plt.subplot(122)
    cax = plot.matshow(info.squeeze())
    plt.colorbar(cax)
    plt.title("Informative sum coeficients")
    plt.show()
    """
    nx, ny, nz = shape
    if nx < 5 or ny < 5:
        raise ValueError("Shape too small minimun is (5, 5, 0)")
    s_obj = np.max([1, np.floor(np.max(shape) / 7)])
    k = 1
    c1 = np.floor((k * nx / 4., ny / 4., nz / 2.))
    d1 = Dot(center=c1, size=s_obj, shape=shape, beta=beta,
             std=std)
    c2 = np.floor((k * nx / 4., ny - (ny / 4.), nz / 2.))
    d2 = Dot(center=c2, size=s_obj, shape=shape, beta=-beta,
             std=0)
    union1 = ObjImage(mask=d1.get_mask() + d2.get_mask(), beta=0,
              std=std)
    k = 3
    c4 = np.floor((k * nx / 4., ny / 4., nz / 2.))
    d4 = Dot(center=c4, size=s_obj, shape=shape, beta=beta,
             std=std)
    c5 = np.floor((k * nx / 4., ny - (ny / 4.), nz / 2.))
    d5 = Dot(center=c5, size=s_obj, shape=shape, beta=-beta,
             std=0)
    union2 = ObjImage(mask=d4.get_mask() + d5.get_mask(), beta=0,
             std=std)
    ## dot in the middle
    c3 = np.floor((nx / 2., ny / 2., nz / 2.))
    d3 = Dot(center=c3, size=s_obj, shape=shape, beta=beta,
             std=std * 2)
    return [d1, d2, union1, d4, d5, union2, d3]