# -*- coding: utf-8 -*-

import math
import numpy as np

'''
In this file different functions to handle the properties of vectors are defined:
- magnitude: the norm of the vector
- distance: the distance between two points
- relevant_position: deals the boundaries of the domain
- dot: scalar multiplication between two vectors
- angle_between: calculates the angle between two vectors
- limit_magnitude: limits the magnitude of a vector given some constraints
'''

def magnitude(x, y):
    return math.sqrt((x ** 2) + (y ** 2))

def distance(x, y):
    distance = magnitude(x[0]-y[0], x[1]-y[1])
    return distance

def relevant_position(position_1, position_2, domain):
    """ Will calculate the relevant position of position_2 regarding position_1
        considering that the edges of the domain are connected. The relevant
        position only takes the distance into account. """
    # distances = [] # distances regarding the domain, and the virutal domains
    position_1 = np.asarray(position_1)
    position_2 = np.asarray(position_2)
    distance_domain = np.linalg.norm(position_1 - position_2)
    dist_virtual = []
    position_v1 = position_2 + np.array([domain[0],0])
    position_v2 = position_2 + np.array([0,domain[0]])
    position_v3 = position_2 + np.array([-domain[0],0])
    position_v4 = position_2 + np.array([0,-domain[1]])
    position_virtual = [position_v1, position_v2, position_v3, position_v4]

    for i in range(4):
        dist_virtual.append(np.linalg.norm(position_1 - position_virtual[i]))
    dist_virtual_array = np.asarray(dist_virtual)
    arg_min = np.argmin(dist_virtual_array)

    if dist_virtual_array[arg_min] < distance_domain:
        position = position_virtual[arg_min]

    else:
        position = position_2

    return position

def dot(a, b):
    return sum(i * j for i, j in zip(a, b))

def angle_between(a, b):
    arg = dot(a, b) / (magnitude(*a) * magnitude(*b))
    if arg > 1:
        arg = 1
    elif arg < -1:
        arg = -1
    angle = math.degrees(math.acos(arg))
    return angle

def limit_magnitude(vector, max_magnitude, min_magnitude = 0.0):
    mag = magnitude(*vector)
    if mag > max_magnitude:
        normalizing_factor = max_magnitude / mag
    elif mag < min_magnitude:
        normalizing_factor = min_magnitude / mag
    else: return vector

    return [value * normalizing_factor for value in vector]