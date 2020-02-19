import cv2
import numpy as np
from PIL import Image
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import matplotlib.pyplot as plt
eye_padding_ratio = 1.2


def slice_eye(img, parts):
    left = min(parts[:, 0])
    top = min(parts[:, 1])
    right = max(parts[:, 0])
    bottom = max(parts[:, 1])
    size, middle_x, middle_y = 0, 0, 0
    if right - left >= top - bottom:
        size = right - left
    else:
        size = top - bottom
    if size % 2 == 1:
        size += 1
    size += 6
    ratio = 60. / 36.
    middle_y = int((top + bottom) / 2)
    middle_x = int((left + right) / 2)
    bottom = int(middle_y + size / ratio / 2 * eye_padding_ratio)
    top = int(middle_y - size / ratio / 2 * eye_padding_ratio)
    left = int(middle_x - size / 2 * eye_padding_ratio)
    right = int(middle_x + size / 2 * eye_padding_ratio)
    result = img[top:bottom, left:right]
    return result, left, top


def find_eye_middle(nparray):
    eye_borders = nparray[(36, 39, 42, 45), :].astype(dtype=np.float32)
    a = sum(eye_borders[:, 0])
    b = sum(eye_borders[:, 1])
    mid_x = a / 4
    mid_y = b / 4
    """[x1, y1] = eye_borders[0]
    [x2, y2] = eye_borders[3]
    k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    cumul = 0
    for [x, y] in eye_borders:
        cumul += ((k * x + b) - y) ** 2
    return (1, k), np.array([mid_x, mid_y])"""
    eye_borders -= np.array([mid_x, mid_y] * 4).reshape((4, 2))
    x = eye_borders[:, 0]
    y = eye_borders[:, 1]
    k = sum(x * y) / sum(x * x)
    vect = np.array([1, k])
    return vect, np.array([mid_x, mid_y])


PADDING = 10


def process_eye(face, np_points):
    solver = pnp_solver.PoseEstimator((720, 1080))
    faceModel = np.array([[-4.50967681e+01, -4.83773045e-01, 2.39702984e+00],
                          [-2.13128582e+01, 4.83773045e-01, -2.39702984e+00],
                          [2.13128582e+01, 4.83773045e-01, -2.39702984e+00],
                          [4.50967681e+01, -4.83773045e-01, 2.39702984e+00],
                          [-2.62995769e+01, 6.85950353e+01, -9.86076132e-32],
                          [2.62995769e+01, 6.85950353e+01, -9.86076132e-32],
                         ])
    headpose_hr, headpose_ht = solver.solve_pose(np_points[(36, 39, 42, 45, 48, 54),], faceModel)
    headpose_ht = headpose_ht.reshape((-1,))
    hR, _ = cv2.Rodrigues(headpose_hr)
    matrix = np.zeros((3, 4))
    matrix[:, :3] = hR
    matrix[:, 3] = headpose_ht
    Fc = []

   # faceModel += np.array([0., -41., 0])
    for i in range(6):
        Fc.append(np.matmul(matrix, [*(faceModel[i]), 1]))
    Fc = np.array(Fc)
    right_eye_center = 0.5 * (Fc[0] + Fc[1])
    left_eye_center = 0.5 * (Fc[2] + Fc[3])
    eye_image_width = 36
    eye_image_height = 60
    eye_one = normalizeImg(face, right_eye_center, hR, [eye_image_width, eye_image_height],
                           solver.camera_matrix)
    eye_two = normalizeImg(face, left_eye_center, hR, [eye_image_width, eye_image_height],
                           solver.camera_matrix)
    eye_one = eye_one.astype(dtype=np.float32).reshape((36, 60, 1)) / 255
    eye_two = eye_two.astype(dtype=np.float32).reshape((36, 60, 1)) / 255
    eye_one = eye_one[::-1]
    eye_two = eye_two[::-1]
    return eye_two, eye_one


def norm(target_3D):
    return np.sqrt(np.matmul(target_3D, target_3D))


def normalizeImg(inputImg, target_3D, hR, roiSize, cameraMatrix, focal_new=960, distance_new=600):
    distance = np.sqrt(np.matmul(target_3D, target_3D))
    z_scale = distance_new / distance
    cam_new = np.array([[focal_new, 0, roiSize[1] / 2], [0.0, focal_new, roiSize[0] / 2], [0, 0, 1.0]])
    scaleMat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]]
    hRx = hR[:, 0]
    forward = (target_3D / distance)
    down = np.cross(forward, hRx)
    down = down / norm(down)
    right = np.cross(down, forward)
    right = right / norm(right)
    rotMat = np.array([right, down, forward])
    warpMat = np.matmul(np.matmul(cam_new, scaleMat), np.matmul(rotMat, np.linalg.inv(cameraMatrix)))
    roiSize = np.array(roiSize, dtype=np.float32)
    warpMat = np.array(warpMat, dtype=np.float32)
    inputImg = np.array(inputImg)
    img_warped = cv2.warpPerspective(inputImg, warpMat, (60, 36))
    return img_warped
