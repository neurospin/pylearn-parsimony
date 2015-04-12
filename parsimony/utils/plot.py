# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:13:42 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay, Tommy Löfstedt
@email:   edouard.duchesnay@cea.fr, lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np
import scipy.stats as ss

__all__ = ["plot_map2d", "plot_map2d_of_models", "plot_classes"]

COLORS = ["b", "g", "r", "c", "m", "y", "k", "w"]
COLORS_FULL = ["blue", "green", "red", "cyan", "magenta", "yellow", "black",
               "white"]
MARKERS = ["+", ".", "o", "*", "p", "s", "x", "D", "h", "^"]
LINE_STYLE = ["-", "--", "--.", ":"]


def plot_map2d(map2d, plot=None, title=None, limits=None,
               center_cmap=True):
    import matplotlib.pyplot as plt

    if plot is None:
        plot = plt

    map2d = map2d.squeeze()

    if len(map2d.shape) != 2:
        raise ValueError("input map is not 2D")

    if np.asarray(limits).size is 2:
        mx = limits[0]
        mi = limits[1]
    else:
        mx = map2d.max()
        mi = map2d.min()

    if center_cmap:
        mx = np.abs([mi, mx]).max()
        mi = -mx

    cax = plot.matshow(map2d, cmap=plt.cm.coolwarm)
    frame = plt.gca()
    frame.get_xaxis().set_visible(False)
    frame.get_yaxis().set_visible(False)
    #k = 1
    #while (10 ** k * mx) < 1 and k < 10:
    #    k += 1
    #ticks = np.array([-mi, -mi / 4 - mi / 2, 0, mx / 2, mx / 2,
    #                  mx]).round(k + 2)
    cbar = plt.colorbar(cax)  # , ticks=ticks)
    cbar.set_clim(vmin=mi, vmax=mx)

    if title is not None:
        plt.title(title)

def plot_map2d_of_models(models_dict, nrow, ncol, shape, title_attr=None):
    """Plot 2 weight maps of models"""
    #from .plot import plot_map2d
    import matplotlib.pyplot as plt
    ax_i = 1
    for k in models_dict.keys():
        mod = models_dict[k]
        if  hasattr(mod, "beta"):
            w = mod.beta
        elif hasattr(mod, "coef_"): # to work with sklean
            w = mod.coef_
        if  (hasattr(mod, "penalty_start") and mod.penalty_start != 0):
            w = w[mod.penalty_start:]
        if (title_attr is not None and hasattr(mod, title_attr)):
            title = getattr(mod, title_attr)
        else:
            title = None
        ax = plt.subplot(nrow, ncol, ax_i)
        plot_map2d(w.reshape(shape), ax,
                   title=title)
        ax_i += 1
    plt.show()

def plot_classes(X, classes, title=None, xlabel=None, ylabel=None, show=True):

    import matplotlib.pyplot as plot

    if isinstance(classes, np.ndarray):
        classes = classes.ravel().tolist()

    cls = list(set(classes))

    # TODO: Add the other cases.
    if X.shape[1] == 2:

        for i in xrange(len(cls)):
            c = cls[i]
            cl = np.array(classes) == c
#            print cl.shape
#            print X[cl, 0].shape
#            print X[cl, 1].shape
            plot.plot(X[cl, 0], X[cl, 1],
                      color=COLORS[i % len(COLORS)],
                      marker='.',
                      markersize=15,
                      linestyle="None")

        if title is not None:
            plot.title(title, fontsize=22)

        if xlabel is not None:
            plot.xlabel(xlabel, fontsize=16)
        if ylabel is not None:
            plot.ylabel(ylabel, fontsize=16)

        if show:
            plot.show()


def plot_errorbars(X, classes=None, means=None, alpha=0.05,
                   title=None, xlabel=None, ylabel=None,
                   colors=None,
                   show=True, latex=True):

    import matplotlib.pyplot as plot

    B, n = X.shape
    if classes is None:
        classes = np.array([1] * n)
    classes = np.array(classes).reshape((n, 1))

    if colors is None:
        colors = COLORS

    data_mu = np.mean(X, axis=0)
    data_df = np.array([B - 1] * n)
    data_sd = np.std(X, axis=0)

    x = np.arange(1, n + 1)

    labels, cls_inverse = np.unique(classes, return_inverse=True)
    labels = labels.ravel().tolist()

#    plot.figure()
    if latex:
        plot.rc('text', usetex=True)
        plot.rc('font', family='serif')
    if means is not None:
        plot.plot(x, means, '*',
                  markerfacecolor="black", markeredgecolor="black",
                  markersize=10)

    ci = ss.t.ppf(1.0 - alpha / 2.0, data_df) * data_sd / np.sqrt(B)

    for i in xrange(len(labels)):
        ind = np.where(classes == labels[i])[0]

        plot.errorbar(x[ind],
                      data_mu[ind],
                      yerr=ci[ind],
                      fmt='o' + colors[i % len(colors)],
                      color=colors[i % len(colors)],
                      ecolor=colors[i % len(colors)],
                      elinewidth=2,
                      markeredgewidth=2,
                      markeredgecolor=colors[i % len(colors)],
                      capsize=5)

    plot.xlim((0, n + 1))
    mn = np.min(data_mu - ci)
    mx = np.max(data_mu + ci)
    d = mx - mn
    plot.ylim((mn - d * 0.05, mx + d * 0.05))

    if xlabel is not None:
        plot.xlabel(xlabel)
    if ylabel is not None:
        plot.ylabel(ylabel)
    if title is not None:
        plot.title(title)
    if show:
        plot.show()