#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:43:37 2022

@author: rigi
"""
import numpy as np

mat = [ 3.9877559882561974e+02, 0., 3.2546639260681752e+02, 0.,
       3.9651789974056226e+02, 2.3906162259594086e+02, 0., 0., 1. ]
f = np.array([mat[0], mat[4]])
c = np.array([mat[2], mat[5]])

# point = np.array([325,239])
point = np.array([320,240])

z = 30

real = ((point-c)/f) * z
print(real)

test = np.array([point[0]/point[1], 1])
print(test*(-z))