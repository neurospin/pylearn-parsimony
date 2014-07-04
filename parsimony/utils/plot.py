# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:13:42 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["plot_map2d", "plot_classes"]

COLORS = ["b", "g", "r", "c", "m", "y", "k", "w"]
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