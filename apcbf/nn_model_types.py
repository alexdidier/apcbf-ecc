""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module specifies all the neural network model types used"""
from enum import Enum

# TODO change s.t. use torch objects

class LinModelType(Enum):
    A = 1
    APLUS = 2
    B = 3
    BPLUS = 4
    C = 5
    CPLUS = 6


class NonLinModelType(Enum):
    A = 1
    APLUS = 2
    B = 3
    BPLUS = 4
    C = 5
    CPLUS = 6
    D = 7
    DPLUS = 8
    EPLUS = 9
    FPLUS = 10
    GPLUS = 11
    TWOHIDDENLAYERS = 20
    THREEHIDDENLAYERS = 21
    FOURHIDDENLAYERS = 22
