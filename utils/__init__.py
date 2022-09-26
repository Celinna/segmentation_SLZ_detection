#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:33:00 2022

@author: rigi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

print(this_dir)
lib_path = osp.join(this_dir, '..', 'models')
add_path(lib_path)