# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:29:42 2015

Copyright (c) 2013-2015, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author: edouard.duchesnay@cea.fr
"""
import numpy as np

__all__ = ['cylinder']


def cylinder(width, nangles):
    """
    Return a mesh of a cylinder that match the topology of a 2D grid.
    Map axis 0 (columns) with "nangles" circular points.
    Map axis 1 (rows) with "width" points along the cylinder width.

    Parameters
    ----------
    width: Integer. Width (number of points) of the cylinder.
    Matches the j (axis 1) of a 2d matrix.

    nangles: Integer. The number of points along the circle.
    Matches the i (axis 0) of a 2d matrix.

    Returns
    -------
    xyz: Numpy array of shape [width * nangles, 3]. The xyz coordinates of the
    mesh vertices. There are width * nangles vertices organized as "nangles"
    blocks of "width" points.

    tri: Numpy array of shape [?, 3]. The triangles linking the vertices

    Example
    -------
    >>> from parsimony.utils import mesh
    >>> xyz, tri = mesh.cylinder(width=5, nangles=10)
    >>> print((int(xyz.shape[0]), int(xyz.shape[1])))
    (50, 3)
    >>> print((int(tri.shape[0]), int(tri.shape[1])))
    (80, 3)
    """
    scale = width / 5.
    angles = np.linspace(0, 2 * np.pi, nangles)
    # 2dgrid to pint index
    xyz = list()
    map_2dgrid_to_ptdx = np.arange(nangles * width).reshape((nangles, width))
    # for each angle iterate over width
    for a in angles:
        for y in np.arange(width):
            xyz.append([np.cos(a) * scale, y, np.sin(a) * scale])

    xyz = np.array(xyz)

    # build triangles
    tri = list()
    for i in range(nangles):
        for j in range(width):
            if (j+1 < width):  # tri bellow (clockwise)
                tri.append([map_2dgrid_to_ptdx[i, j],
                            map_2dgrid_to_ptdx[i, j+1],
                            map_2dgrid_to_ptdx[(i+1) % nangles, j+1]])
            if (j+1 < width):  # tri above (clockwise)
                tri.append([map_2dgrid_to_ptdx[i, j],
                            map_2dgrid_to_ptdx[i-1, j],
                            map_2dgrid_to_ptdx[i, j+1]])

    tri = np.array(tri)

    return xyz, tri

if __name__ == "__main__":
    import unittest
    unittest.main()
