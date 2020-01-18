import copy
from math import sqrt
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from predictor import normalize


def get_normal_vector(vector0, vector1):
    res = np.cross(vector0, vector1)
    res = normalize(res)
    return res


def get_surface_by_norm_and_point(norm, point):
    a, b, c = norm
    d = -np.matmul([a, b, c], point)
    return [a, b, c, d]


def get_distance_surface_point(surface, point):
    normal = surface[:3]
    return abs(np.matmul(surface, [*point, 1])) / sqrt(np.matmul(normal, normal))


def get_r_coef(surface, point, norm):
    assert (norm - normalize(norm) < [0.001, 0.001, 0.001]).all()
    r = get_distance_surface_point(surface, point)
    point_in_surface = point + r * norm
    if np.matmul([*point_in_surface, 1], surface) != 0:
        r = r * -1
    return r


def get_point_between_lines(point0, vector0, point1, vector1):
    norm = get_normal_vector(vector0, vector1)
    surface = get_surface_by_norm_and_point(norm, point0)
    r = get_r_coef(surface, point1, norm)
    a_0, b_0, c_0 = point0
    a_1, b_1, c_1 = point1
    x_0, y_0, z_0 = vector0
    x_1, y_1, z_1 = vector1
    x_n, y_n, z_n = norm
    t = -(-a_0 * y_1 + a_1 * y_1 + b_0 * x_1 - b_1 * x_1 - r * x_1 * y_n + r * x_n * y_1) / (x_1 * y_0 - x_0 * y_1)
    g = -(-a_0 * y_0 + a_1 * y_0 + b_0 * x_0 - b_1 * x_0 - r * x_0 * y_n + r * x_n * y_0) / (x_1 * y_0 - x_0 * y_1)
    return (np.array(point0) + t * np.array(vector0) + np.array(point1) + g * np.array(vector1)) / 2


def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


def minimize(point_array, starting_surface):
    points = copy.deepcopy(point_array)

    def function(x):
        fit = 0
        for point in points:
            fit += np.matmul([*point, 1], x) ** 2
        fit /= x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
        return fit

    x0 = starting_surface
    res = scipy.optimize.minimize(function, x0, method='BFGS', options={'disp': False, 'gtol': 1e-8})
    return res


def minimize_svd(point_array):
    mean = np.zeros((3,), dtype=np.float64)
    for point in point_array:
        mean += point
    mean /= point_array.shape[0]
    cur_point = copy.deepcopy(point_array)
    for point in cur_point:
        point -= mean
    u, s, vh = np.linalg.svd(cur_point, full_matrices=False)
    u = u
    x = vh[2]
    d = - np.matmul(x, mean)
    return [*x, d]


def function(x, points):
    fit = 0
    for point in points:
        fit += np.matmul([*point, 1], x) ** 2
    fit /= x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    return fit


def draw_plot(plane, axis, min_x=-200, min_y=-200, max_x=205, max_y=205):
    a, b, c, d = plane
    xx, yy = np.meshgrid(range(min_x, max_x+5, max_x-min_x), range(min_y, max_y+5, max_y-min_y))
    z = -(a * xx + b * yy + d) / c
    axis.plot_surface(xx, yy, z)


def kek(points):
    # create random data
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    # plot raw data

    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    # plot plane
    return [np.array(fit[0])[0][0], np.array(fit[1])[0][0], -1, np.array(fit[2])[0][0]]


def plane_by_3_points(p1, p2, p3):
    vect1 = p1 - p2
    vect2 = p1 - p3
    a, b, c = np.cross(vect1, vect2)
    d = -np.matmul([a, b, c], p1)
    return [a, b, c, d]


def get_plane_by_eye_vectors(screen_points):
    z = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#ff0000', '#00ff00', '#0000ff', '#000000', '#ffffff']
    for screen_point, color in zip(screen_points, colors):
        z.append(get_point_between_lines(*screen_point['left'], *screen_point['right']))
        arr = np.array(screen_point['left'])
        arr[1] = arr[1] * 1000 + arr[0]
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=color)
        arr = np.array(screen_point['right'])
        arr[1] = arr[1] * 1000 + arr[0]
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=color)
    z = np.array(z)
    ax.scatter(z[:, 0], z[:, 1], z[:, 2])
    res1 = minimize_svd(z)
    res2 = minimize(z, plane_by_3_points(z[0], z[2], z[-1]))
    res2 = res2['x']
    res3 = kek(z)
    val1 = function(res1, z)
    val2 = function(res2, z)
    val3 = function(res3, z)
    res = np.array([res1, res2, res3])[np.argmin([val1, val2, val3])]
    draw_plot(res, ax)
    plt.show()
    return res
