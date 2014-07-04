# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:21:44 2013

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Jinpeng Li
@email:   jinpeng.li@cea.fr
@license: BSD 3-clause.
"""
import numpy as np


class SpamsGenerator:

    def get_x_y_estimated_beta(self):
        """
        Reference:
        ---------
        http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23
        """
        shape = (4, 4, 1)
        num_samples = 10
        coefficient = 0.05

        num_ft = shape[0] * shape[1] * shape[2]
        X = np.random.random((num_samples, num_ft))
        beta = np.random.random((num_ft, 1))
        # y = dot(X, beta) + noise
        y = np.dot(X, beta) + np.random.random((num_samples, 1)) * 0.0001

        try:
            import spams
            # Normalization for X
            X = np.asfortranarray(X)
            X = np.asfortranarray(X - np.tile(
                                  np.mean(X, 0),
                                  (X.shape[0], 1)))
            X = spams.normalize(X)
            # Normalization for y
            y = np.asfortranarray(y)
            y = np.asfortranarray(y - np.tile(
                                  np.mean(y, 0),
                                  (y.shape[0], 1)))
            y = spams.normalize(y)
            weight0 = np.zeros((X.shape[1], y.shape[1]),
                               dtype=np.float64,
                               order="FORTRAN")
            param = {'numThreads': 1, 'verbose': True,
                 'lambda1': coefficient, 'it0': 10, 'max_it': 200,
                 'L0': 0.1, 'tol': 1e-3, 'intercept': False,
                 'pos': False}
            param['compute_gram'] = True
            param['loss'] = 'square'
            param['regul'] = 'l2'
            (weight_ridge, optim_info) = spams.fistaFlat(y,
                                                  X,
                                                  weight0,
                                                  True,
                                                  **param)
            param['regul'] = 'l1'
            (weight_l1, optim_info) = spams.fistaFlat(y,
                                                 X,
                                                 weight0,
                                                 True,
                                                 **param)
#            print "X = ", repr(X)
#            print "y = ", repr(y)
#            print "weight_ridge =", repr(weight_ridge)
#            print "weight_l1 =", repr(weight_l1)
        except ImportError:
            # TODO: Don't use print directly.
            print "Cannot import spams. Default values will be used."
            X = np.asarray([
           [ 0.26856766,  0.30620391,  0.26995615,  0.3806023 ,  0.41311465,
            -0.24685479,  0.34108499, -0.22786788, -0.2267594 ,  0.30325884,
            -0.00382229,  0.3503643 ,  0.21786749, -0.15275043, -0.24074157,
            -0.25639825],
           [-0.14305316, -0.19553497,  0.45250255, -0.17317269, -0.00304901,
             0.43838073,  0.01606735,  0.09267714,  0.47763275,  0.23234948,
             0.38694597,  0.72591941,  0.21028899,  0.42317021,  0.276003  ,
             0.42198486],
           [-0.08738645,  0.10795947,  0.45813373, -0.34232048,  0.43621128,
            -0.36984753,  0.16555311,  0.55188325, -0.48169657, -0.52844883,
             0.15140672,  0.06074575, -0.36873621,  0.23679974,  0.47195386,
            -0.09728514],
           [ 0.16461237,  0.30299873, -0.32108348, -0.53918274,  0.02287831,
             0.01105383, -0.11124968,  0.18629018,  0.30017151, -0.04217922,
            -0.46066699, -0.33612491, -0.52611772, -0.25397362, -0.27198468,
            -0.42883518],
           [ 0.4710195 ,  0.35047152, -0.07990029,  0.34911632,  0.07206932,
            -0.20270895, -0.0684226 , -0.18958745, -0.08433092,  0.14453963,
             0.28095469, -0.35894296,  0.11680455, -0.37598039, -0.28331446,
            -0.00825299],
           [-0.420528  , -0.74469306,  0.22732681,  0.34362884,  0.16006124,
            -0.29691759,  0.27029047, -0.31077084, -0.048071  ,  0.36495065,
             0.49364453, -0.16903801,  0.07577839, -0.36492748,  0.09448284,
            -0.37055486],
           [ 0.4232946 , -0.26373387, -0.01430445, -0.2353587 , -0.5005603 ,
            -0.35899458,  0.32702596, -0.38311949,  0.31862621, -0.31931012,
            -0.41836583, -0.02855145, -0.50315227, -0.34807958, -0.05252361,
             0.11551424],
           [-0.28443208,  0.07677476, -0.23720305,  0.11056299, -0.48742565,
             0.36772457, -0.56074202,  0.3145033 , -0.22811763,  0.36482173,
            -0.01786535, -0.02929555,  0.35635411,  0.45838473,  0.45853286,
             0.00159594],
           [-0.45779277,  0.10020579, -0.30873257,  0.28114072,  0.18120182,
             0.33333004,  0.17928387,  0.31572323,  0.32902088, -0.10396976,
            -0.33296829,  0.05277326,  0.27139148,  0.18653329,  0.06068255,
            -0.01942451],
           [ 0.06569833, -0.04065228, -0.44669538, -0.17501657, -0.29450165,
             0.32483427, -0.55889145, -0.34973144, -0.35647584, -0.41601239,
            -0.07926316, -0.26784983,  0.14952119,  0.19082353, -0.51309079,
             0.6416559 ]])
            y = np.asarray([
               [ 0.15809895],
               [ 0.69496971],
               [ 0.01214928],
               [-0.39826324],
               [-0.01682498],
               [-0.03372654],
               [-0.45148804],
               [ 0.21735376],
               [ 0.08795349],
               [-0.27022239]])
            weight_ridge = np.asarray([
               [ 0.038558  ],
               [ 0.12605106],
               [ 0.19115798],
               [ 0.07187217],
               [ 0.09472713],
               [ 0.14943554],
               [-0.01968095],
               [ 0.11695959],
               [ 0.15049031],
               [ 0.18930644],
               [ 0.26086626],
               [ 0.23243305],
               [ 0.17425178],
               [ 0.13200238],
               [ 0.11710994],
               [ 0.11272092]])
            weight_l1 = np.asarray([
               [ 0.        ],
               [ 0.02664519],
               [ 0.        ],
               [ 0.        ],
               [ 0.        ],
               [ 0.10357106],
               [ 0.        ],
               [ 0.2103012 ],
               [ 0.00399881],
               [ 0.10815184],
               [ 0.32221254],
               [ 0.49350083],
               [ 0.21351531],
               [ 0.        ],
               [ 0.        ],
               [ 0.        ]])

        ret_data = {}
        ret_data['X'] = X
        ret_data['y'] = y
        ret_data['weight_ridge'] = weight_ridge
        ret_data['weight_l1'] = weight_l1
        ret_data['coefficient'] = coefficient
        ret_data['shape'] = shape
        ret_data['num_samples'] = num_samples
        ret_data['num_ft'] = num_ft

        return ret_data

if __name__ == "__main__":
    spams_generator = SpamsGenerator()
    print spams_generator.get_x_y_estimated_beta()