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

__all__ = ["map2d", "map2d_of_models", "classes", "errorbars",
           "voronoi_tesselation"]


COLORS = ["b", "g", "r", "c", "m", "y", "k", "w"]
COLORS_FULL = ["blue", "green", "red", "cyan", "magenta", "yellow", "black",
               "white"]
MARKERS = ["+", ".", "o", "*", "p", "s", "x", "D", "h", "^"]
LINE_STYLE = ["-", "--", "--.", ":"]


def map2d(map2d, plot=None, title=None, limits=None, center_cmap=True):

    import matplotlib.pyplot as plt

    if plot is None:
        plot = plt

    map2d = map2d.squeeze()

    if len(map2d.shape) != 2:
        raise ValueError("input map is not 2D")

    if np.asarray(limits).size == 2:
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
    # k = 1
    # while (10 ** k * mx) < 1 and k < 10:
    #     k += 1
    # ticks = np.array([-mi, -mi / 4 - mi / 2, 0, mx / 2, mx / 2,
    #                   mx]).round(k + 2)
    cbar = plt.colorbar(cax)  # , ticks=ticks)
    if hasattr(cbar, "set_clim"):
        cbar.set_clim(vmin=mi, vmax=mx)
    else:
        cbar.mappable.set_clim(vmin=mi, vmax=mx)

    if title is not None:
        plt.title(title)


def map2d_of_models(models_dict, nrow, ncol, shape, title_attr=None):
    """Plot 2 weight maps of models."""
    import matplotlib.pyplot as plt

    ax_i = 1
    for k in list(models_dict.keys()):
        mod = models_dict[k]
        if hasattr(mod, "beta"):
            w = mod.beta
        elif hasattr(mod, "coef_"):  # to work with sklean
            w = mod.coef_
        if (hasattr(mod, "penalty_start") and mod.penalty_start != 0):
            w = w[mod.penalty_start:]
        if (title_attr is not None and hasattr(mod, title_attr)):
            title = getattr(mod, title_attr)
        else:
            title = None
        ax = plt.subplot(nrow, ncol, ax_i)
        map2d(w.reshape(shape), ax, title=title)
        ax_i += 1
    plt.show()


def classes(X, classes, title=None, xlabel=None, ylabel=None, show=True):

    import matplotlib.pyplot as plt

    if isinstance(classes, np.ndarray):
        classes = classes.ravel().tolist()

    cls = list(set(classes))

    # TODO: Add the other cases.
    if X.shape[1] == 2:

        for i in range(len(cls)):
            c = cls[i]
            cl = np.array(classes) == c
#            print cl.shape
#            print X[cl, 0].shape
#            print X[cl, 1].shape
            plt.plot(X[cl, 0], X[cl, 1],
                     color=COLORS[i % len(COLORS)],
                     marker='.',
                     markersize=15,
                     linestyle="None")

        if title is not None:
            plt.title(title, fontsize=22)

        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=16)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=16)

        if show:
            plt.show()


def errorbars(X, classes=None, means=None, alpha=0.05,
              title=None, xlabel=None, ylabel=None, colors=None,
              new_figure=True, show=True, latex=True, ylim=None):

    import matplotlib.pyplot as plt

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

    if new_figure:
        plt.figure()
    if latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    if means is not None:
        plt.plot(x, means, '*',
                 markerfacecolor="black", markeredgecolor="black",
                 markersize=10)

    ci = ss.t.ppf(1.0 - alpha / 2.0, data_df) * data_sd / np.sqrt(B)

    for i in range(len(labels)):
        ind = np.where(classes == labels[i])[0]

        plt.errorbar(x[ind],
                     data_mu[ind],
                     yerr=ci[ind],
                     fmt='o' + colors[i % len(colors)],
                     color=colors[i % len(colors)],
                     ecolor=colors[i % len(colors)],
                     elinewidth=2,
                     markeredgewidth=2,
                     markeredgecolor=colors[i % len(colors)],
                     capsize=5)

    plt.xlim((0, n + 1))
    if ylim is not None:
        plt.ylim(ylim)
    else:
        mn = np.min(data_mu - ci)
        mx = np.max(data_mu + ci)
        d = mx - mn
        plt.ylim((mn - d * 0.05, mx + d * 0.05))

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def voronoi_tesselation(mu, rect=None, nx=100, ny=100,
                        colors=None, alpha=0.3, show=True,
                        markersize=10):

    import matplotlib.pyplot as plt

    mu = np.array(mu)  # A 2D data structure -> numpy array

    from scipy.spatial import Voronoi

    vor = Voronoi(mu)  # Compute Voronoi tesselation.
    regions, vertices = _voronoi_finite_polygons_2d(vor)

    if colors is None:
        colors = [[0.0, 0.0, 0.0]] * mu.shape[0]
        for i in range(len(colors)):
            colors[i] = [np.random.rand(),
                         np.random.rand(),
                         np.random.rand()]

    for i in range(len(regions)):
        region = regions[i]
        polygon = vertices[region]
        plt.fill(*list(zip(*polygon)), alpha=alpha, color=colors[i])
        plt.plot(mu[i, 0], mu[i, 1], marker='o', color=colors[i],
                 markersize=markersize)

    if rect is None:
        plt.xlim(vor.min_bound[0] * 0.9, vor.max_bound[0] * 1.1)
        plt.ylim(vor.min_bound[1] * 0.9, vor.max_bound[1] * 1.1)
    else:
        plt.xlim(rect[0], rect[1])
        plt.ylim(rect[2], rect[3])

    if show:
        plt.show()


def _voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite voronoi regions in a 2D diagram to finite regions.

    This code is adapted from: https://gist.github.com/pv/8036995

    Parameters
    ----------
    vor : Voronoi. Input diagram.

    radius : float, optional. Distance to "points at infinity".

    Returns
    -------
    regions : List of tuples. Indices of vertices in each revised Voronoi
            regions.

    vertices : List of tuples. Coordinates for revised Voronoi vertices. Same
            as coordinates of input vertices, with "points at infinity"
            appended to the end.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 1000.0

    # Construct a map containing all ridges for a given point.
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions.
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region.
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region.
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region.
                continue

            # Compute the missing endpoint of an infinite ridge.
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # Finish.
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
