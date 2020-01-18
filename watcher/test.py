import copy
from math import sqrt, cos, sin
from random import randint, random, uniform
import random

import matplotlib.pyplot as plt
import numpy as np
# import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import os
from mpl_toolkits.mplot3d import Axes3D
import scipy

from predictor_module import normalize





if __name__ == "__main__":
    # TODO TEST AND IMPLEMENT
    result = [0, 0, 0]
    for i in range(1000):
        x, y = np.meshgrid(range(-20, 21, 10)), np.meshgrid(range(-20, 21, 10))
        kx, ky = random.uniform(-1000, 1000), random.uniform(-1000, 1000)
        z = []
        z_level = random.uniform(-1000, 1000)
        for a in x[0]:
            for b in y[0]:
                z.append([a, b, kx*a+ky*b+np.random.normal(0, 10) - z_level])
        z = np.array(z)

    print(result)

    """for i in range(100):
        x = [[1, 0, 1], [1, 1, 0], [0, 1, 3], [0, 1, 0]]
        to_random_coord_translator = rvs(dim=3)
        for i in range(4):
            x[i] = np.matmul(to_random_coord_translator, x[i])
        from_random = np.linalg.inv(to_random_coord_translator)
        res = np.matmul(from_random, get_point_between_lines(*x))"""
