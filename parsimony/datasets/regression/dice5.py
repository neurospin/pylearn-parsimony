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
from parsimony.datasets.utils import Dot, ObjImage, spatial_smoothing, corr_to_coef


def load(n_samples=100, shape=(30, 30, 1),
                           r2=.75,
                           sigma_spatial_smoothing=1,
                           signal_std_pixel_obj_ratio=1.,
                           model = "independant",
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

    model:  string or a dict (default "independant")
        If model is "independant":
            # All points has an independant latent
            l1=1., l2=1., l3=1., l4=1., l5=1.,
            # No shared variance:
            l12=0., l45=0., l12345=0.,
            # Five dots contribute equally:
            b1=1., b2=1., b3=1., b4=-1., b5=-1.

        if model is a dictionary:
            update (overwrite) independant model by dictionnary parameter
            Example set betas of points 4 and 5 to 1
            dict(b4=1., b5=1.)

        If model is "redundant":
            # Point-level signal in dots 1 an 2 stem from shared latent:
            l1=0., l2=0., l12 =1.,
            # l3 is independant:
            l3=1.,
            # Point-level signal in dots 4 an 5 stem from shared latent:
            l4=0., l5=0., l45=1.,
            # No global shared variance:
            l12345 = 0.,
            # Five dots contribute equally:
            b1=1., b2=1., b3=1., b4=-1., b5=-1.

        If model is "suppressor":
            # Point-level signal in dot 2 stem only from shared latent:
            l1=1, l2=0., l12=1.,
            # l3 is independant:
            l3 = 1.,
            # Point-level signal in dot 5 stem from shared latent:
            l4=1., l5=0., l45=1.,
            # No global shared variance:
            l12345 = 0.,
            # Dot 2 suppresses shared signal with dot 1, dot 5 suppresses dot 4:
            b1=1., b2=-1., b3=1., b4=1., b5=-1.

            y = X1       - X2  + X3 + X4       - X5  + noise
            y = l1 + l12 - l12 + l3 + l4 + l45 - l45 + noise
            y = l1 + l3 + l4 + noise
            So pixels of X2 and X5 are not correlated with the target y so they
            will not be detected by univariate analysis. However, they
            are usefull since they are suppressing unwilling variance that stem
            from latents l12 and l45.

    signal_std_pixel_obj_ratio: Float. Controls the ratio between object-level signal
            and pixel-level signal for pixels within objects. If
            signal_std_pixel_obj_ratio == 1 then 100% of the signal of pixels within
            the same object is shared (ie.: no pixel level) signal. If
            signal_std_pixel_obj_ratio == 0 then all the signal is pixel specific.
            High signal_std_pixel_obj_ratio promotes spatial correlation between
            pixels of the same object.

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
           - Sample five latent variables ~ N(0, 1): l1, l3, l4, l12, l45, l12345.
           l1: latent (shared variance) for all pixels of point 1.
           ...
           l5: latent (shared variance) for all pixels of point 5.
           l12: latent (shared variance) for all pixels of point 1 and 2.
           l45: latent (shared variance) for all pixels of point 4 and 5 .
           l12345: latent (shared variance) for all pixels of point 1, 2, 3, 4 and 5.

           - Pixel i of dots X1, X2, X3, X4, X5 are sampled as:
             X1i = l1 + l12 + l12345 + Xi
             X2i = l2 + l12 + l12345 + Xi
             X3i = l3 + l12345 + Xi
             X4i = l4 + l45 + l12345 + Xi
             X5i = l5 + l45 + l12345 + Xi
             Note that:
             Pixels of dot X1 share a common variance that stem from l1, l12 and l12345
             Pixels of dot X2 share a common variance that stem from l1, l12 and l12345
             Pixels of dot X1 and pixel of dot X2 share a common variance that
             stem from l12.
             etc.

        4) Spatial Smoothing.

        5) Model: y = X beta + noise
        - Betas are null outside dots, and b1, b2, b3, b4, b5 within dots
        - Sample noise ~ N(0, 1)
        - Compute X beta then scale beta such that: r_squared(y, X beta) = r2
        Return X, y, beta


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
    # TODO use signal_std_pixel_obj_ratio to fix signal_std_obj
    signal_std_pix = signal_std_obj = 1  # items std-dev
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
    X = np.random.normal(mu_e, signal_std_pix, n_samples * n_features)
    X3d = X.reshape(n_samples, nx, ny, nz)
    #########################################################################
    ## 2. Tune points parameters latent and beta
    # Default model independant points
    model_ = dict(
            # All points has an independant latent
            l1=1., l2=1., l3=1., l4=1., l5=1.,
            # No shared variance:
            l12=0., l45=0., l12345=0.,
            # Five dots contribute equally:
            b1=1., b2=1., b3=1., b4=-1., b5=-1.)
    if isinstance(model, dict):
        model_.update(model)
    elif model is "redundant":
        model_ = dict(
            # Point-level signal in dots 1 an 2 stem from shared latent:
            l1=0., l2=0., l12 =1.,
            # l3 is independant:
            l3=1.,
            # Point-level signal in dots 4 an 5 stem from shared latent:
            l4=0., l5=0., l45=1.,
            # No global shared variance:
            l12345 = 0.,
            # Five dots contribute equally:
            b1=1., b2=1., b3=1., b4=-1., b5=-1.)
    elif model is "suppressor":
        model_ = dict(
            # Point-level signal in dot 2 stem only from shared latent:
            l1=1, l2=0., l12=1.,
            # l3 is independant:
            l3=1.,
            # Point-level signal in dot 5 stem from shared latent:
            l4=1., l5=0., l45=1.,
            # No global shared variance:
            l12345 = 0.,
            # Dot 2 suppresses shared signal with dot 1, dot 5 suppresses dot 4:
            b1=1., b2=-1., b3=1., b4=1., b5=-1.)
    model_["l1"] *= signal_std_obj
    model_["l2"] *= signal_std_obj
    model_["l3"] *= signal_std_obj
    model_["l4"] *= signal_std_obj
    model_["l5"] *= signal_std_obj
    model_["l12"] *= signal_std_obj
    model_["l45"] *= signal_std_obj
    model_["l12345"] *= signal_std_obj
    #########################################################################
    ## 3. Build Objects
    d1, d2, d3, d4 ,d5, union12, union45, union12345 = dice_five_with_union_of_pairs(shape)
    d1.std = model_["l1"]
    d1.beta = model_["b1"]
    d2.std = model_["l2"]
    d2.beta = model_["b2"]
    union12.std = model_["l12"]
    union12.beta = 0.
    d4.std = model_["l4"]
    d4.beta = model_["b4"]
    d5.std = model_["l5"]
    d5.beta = model_["b5"]
    union45.std = model_["l45"]
    union45.beta = 0.
    d3.std = model_["l3"]
    d3.beta = model_["b3"]
    union12345.std = model_["l12345"]
    union12345.beta = 0.
    objects = [d1, d2, d3, d4 ,d5, union12, union45, union12345]
    #########################################################################
    ## 3. Object-level structured signal
    X3d, support = ObjImage.object_model(objects, X3d)
    #########################################################################
    ## 4. Pixel-level signal structure: spatial smoothing
    if sigma_spatial_smoothing != 0:
        X3d = spatial_smoothing(X3d, sigma_spatial_smoothing, mu_e,
                                  signal_std_pix)
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
def dice_five_with_union_of_pairs(shape):
    """Seven objects, five dot + union1 = dots 1 + 2 and union2 = dots 4 + 5

    Examples
    --------
    shape = (5, 5, 1)
    map = np.zeros(shape)
    info = np.zeros(shape)
    for o in dice_five_with_union_of_pairs(shape):
       map[o.get_mask()] += 1.
    import matplotlib.pyplot as plt
    plot.matshow(map.squeeze())
    plt.show()
    """
    nx, ny, nz = shape
    if nx < 5 or ny < 5:
        raise ValueError("Shape too small minimun is (5, 5, 0)")
    s_obj = np.max([1, np.floor(np.max(shape) / 7)])
    k = 1
    c1 = np.floor((k * nx / 4., ny / 4., nz / 2.))
    d1 = Dot(center=c1, size=s_obj, shape=shape)
    c2 = np.floor((k * nx / 4., ny - (ny / 4.), nz / 2.))
    d2 = Dot(center=c2, size=s_obj, shape=shape)
    union12 = ObjImage(mask=d1.get_mask() + d2.get_mask())
    k = 3
    c4 = np.floor((k * nx / 4., ny / 4., nz / 2.))
    d4 = Dot(center=c4, size=s_obj, shape=shape)
    c5 = np.floor((k * nx / 4., ny - (ny / 4.), nz / 2.))
    d5 = Dot(center=c5, size=s_obj, shape=shape)
    union45 = ObjImage(mask=d4.get_mask() + d5.get_mask())
    ## dot in the middle
    c3 = np.floor((nx / 2., ny / 2., nz / 2.))
    d3 = Dot(center=c3, size=s_obj, shape=shape)
    union12345 = ObjImage(mask=d1.get_mask() + d2.get_mask() + d3.get_mask() +
        d4.get_mask() + d5.get_mask())
    return [d1, d2, d3, d4 ,d5, union12, union45, union12345]