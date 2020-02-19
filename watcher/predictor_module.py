import collections
import copy
import functools
import json
import subprocess
import threading
import time
from functools import partial
from math import sqrt

import cv2
import dlib
import numpy as np
import pyautogui
import tensorflow as tf
import zerorpc
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import calibrator
import drawer
import eye_module
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import image_translation
import model as model_gen
from from_internet_or_for_from_internet import PNP_solver as pnp_solver


class BasicPredictor:
    def predict_eye_vector_and_face_points(self, imgs, time_now, configurator=None):
        raise NotImplementedError

    def get_mouse_coords(self, cameras, time_now):
        face, results, out_inform = self.predict(cameras, time_now)
        # out_inform["rotator"] -= angle
        # out_inform["cutter"] -= offset
        print(str(out_inform))

        result = list(results)
        results = [0, 0]
        for cam in result:
            for eye in cam:
                results[0] += eye[0]
                results[1] += eye[1]
        results[0] /= len(result) * 2
        results[1] /= len(result) * 2
        if results[0] < 0:
            results[0] = 0
        if results[1] < 0:
            results[1] = 0
        if results[0] >= 1920:
            results[0] = 1919
        if results[1] >= 1080:
            results[1] = 1079
        pyautogui.moveTo(results[0], results[1])

    def predict(self, cameras, time_now, ):
        imgs = []
        for camera in cameras:
            imgs.append(camera.get_picture())

        faces, eye_one_vectors, eye_two_vectors, np_points_all, tmp_out_informs = \
            self.predict_eye_vector_and_face_points(imgs, time_now, )

        results = []
        for i, camera in zip(range(len(cameras)), cameras):
            results.append(
                camera.update_gazes_history(eye_one_vectors[i], eye_two_vectors[i], np_points_all[i], time_now))

        def face_to_img(face):
            if not isinstance(face, Image.Image):
                face = Image.fromarray(face)
            return face

        faces = list(map(lambda face: face_to_img(face), faces))
        faces = image_translation.pack_to_one_image(*faces)
        return faces, results, tmp_out_informs


class GoodPredictor(BasicPredictor):

    def predict_eye_vector_and_face_points(self, imgs, time_now, configurator=None):
        faces = []
        eye_one_vectors = []
        eye_two_vectors = []
        np_points_all = []
        tmp_out_informs = []

        for img in imgs:
            np_points, resizing_cropper, img = detect_face_points_dlib(np.array(img), cropping_needed=False)
            if np_points is not None:
                try:
                    tmp_out_inform = {}
                    face = img
                    #    np_points[i] = rotate_point(np_points[i], middle, -angle)

                    # face_cutter = calibrator.FaceCropper(np_points)
                    # face, np_points = face_cutter.apply_forth(img)
                    # rotator = calibrator.RotationTranslation(np_points)
                    # face, np_points = rotator.apply_forth(img)
                    to_out_face = face
                    # tmp_out_inform["cutter"] = face_cutter.get_modification_data()
                    # tmp_out_inform["rotator"] = rotator.get_modification_data()
                    eyes = []

                    eyes.append(eye_module.process_eye(face, np_points))
                    eyes_to_predict = []
                    for eye_one, eye_two in eyes:
                        eyes_to_predict.append(eye_one)
                        eyes_to_predict.append(eye_two)
                    res = model.predict(np.array(eyes_to_predict))
                    eye_one_vector = normalize(res[0])
                    eye_two_vector = normalize(res[1])
                    face = Image.fromarray(face)
                    drawer = ImageDraw.Draw(face)
                    # #   for [x, y] in np_points:
                    # #   #     drawer.ellipse([x-2, y-2, x+2, y+2], fill=0)
                    eye_middle = np.average(np_points[36:41], axis=0)
                    drawer.ellipse([eye_middle[0] - 2, eye_middle[1] - 2, eye_middle[0] + 2, eye_middle[1] + 2])
                    faces.append(face)
                    rotation, translation = solver.solve_pose(np_points)
                    translation = translation.reshape((-1))
                    rotation, _ = cv2.Rodrigues(rotation)

                    eye_one_vector = rotation @ eye_one_vector
                    eye_two_vector = rotation @ eye_two_vector
                    eye_one_vectors.append(eye_one_vector)
                    eye_two_vectors.append(eye_two_vector)
                    np_points_all.append(np_points)
                    tmp_out_informs.append(tmp_out_inform)

                    """fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1, projection='3d')

                    points_new = []
                    for point in solver.model_points_68:
                        points_new.append(rotation@point + translation.reshape((-1, )))
                    points_new = np.array(points_new)
                    left_eye = sum(points_new[36:42]) / 6
                    right_eye = sum(points_new[42:48]) / 6
                    ax.scatter(*left_eye, color="#ff0000")
                    ax.scatter(*right_eye, color="#ff0000")
                    ax.scatter(*(right_eye + np.array([0, 0, 100])), color="#ff0000")
                    ax.scatter(*(left_eye + eye_one_vector*100), color="#00ff00")
                    ax.scatter(*(right_eye + eye_two_vector * 100), color="#00ff00")
                    points_new = np.array(points_new).transpose()
                    ax.scatter(*points_new)

                    plt.show()"""
                except:
                    pass
        return faces, eye_one_vectors, eye_two_vectors, np_points_all, tmp_out_informs


def detect_face_points_dlib(src, cropping_needed=True):
    img = copy.deepcopy(src)
    cur_time = time.time()
    smaller_img = copy.deepcopy(img)
    if not (isinstance(src, Image.Image)):
        smaller_img = Image.fromarray(smaller_img)
    smaller_img_size = (512, int(512 * smaller_img._size[1] / smaller_img._size[0]))
    smaller_img = smaller_img.resize(smaller_img_size)
    smaller_img = np.array(smaller_img)
    try:
        smaller_dets = detector(smaller_img, 0)
        shape = predictor(smaller_img, smaller_dets[0])
        np_points = np.zeros((shape.num_parts, 2), dtype=np.float32)
        for i in range(shape.num_parts):
            np_points[i, 0] = shape.part(i).x
            np_points[i, 1] = shape.part(i).y
        resizing_cropper = calibrator.FaceCropper(np_points, smaller_img)
        img, _ = resizing_cropper.apply_forth(img)
        img = np.array(img)
        dets = detector(img, 0)
        shape = predictor(img, dets[0])
        np_points = np.zeros((shape.num_parts, 2), dtype=np.float32)
        for i in range(shape.num_parts):
            np_points[i, 0] = shape.part(i).x
            np_points[i, 1] = shape.part(i).y
        if cropping_needed:
            return np_points, resizing_cropper, img
        else:
            offset = list(resizing_cropper.get_resulting_offset())
            # offset[0] *= src.width/smaller_img_size[0]
            # offset[1] *= src.height / smaller_img_size[1]
            for i in range(shape.num_parts):
                np_points[i] += offset
            return np_points, resizing_cropper, src
    except:
        pass


def generic_pixel_loss(y_true, y_pred, pixel_func, sample_weight=0):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    pix_pred = pixel_func(y_pred)
    pix_true = pixel_func(y_true)
    return tf.reduce_mean(tf.sqrt(tf.square(pix_pred - pix_true)))


faces_folder_path = "faces"
predictor_path = "68_predictor.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


@tf.function
def get_pixel(tens, a, b, c, d,
              width=0.54, heigth=0.30375,
              pixel_width=1920, pixel_heigth=1080):
    """k = -d / (a * x + b * y + c * z)
    print(x[0].item())
    x = k * x
    y = k * y
    z = k * z
    pixel_x = x / width * pixel_width
    pixel_y = y / heigth * pixel_heigth
    return pixel_x, pixel_y """
    fl = np.array([a, b, c])
    ratios = np.array([width * pixel_width, heigth * pixel_heigth])
    bottom = tf.math.reduce_sum(tens * fl, 1)
    k = d / (tf.math.reduce_sum(tens * fl, 1))
    k = tf.convert_to_tensor([k, k, k])
    k = tf.transpose(k)
    tens = tens * k
    tens = tens[:, 0:2]
    tens = tens * ratios
    return tens


tf.debugging.set_log_device_placement(True)

model = model_gen.get_model()

model.load_weights("checkpoint_path/")
print(model)
kek = tf.convert_to_tensor([1e-8 for i in range(1)], dtype=tf.float32)
pixel_loss = partial(generic_pixel_loss, pixel_func=partial(get_pixel, a=0, b=0, c=1, d=-1))
pixel_func = partial(get_pixel, a=0, b=0, c=1, d=-1)

history = collections.deque()
calibration_results = np.zeros((4, 2), dtype=np.float32)
calibration_history = []


def normalize(v):
    norm = 0
    for vi in v:
        norm += vi * vi
    norm = sqrt(norm)
    if norm == 0:
        return v
    return v / norm


solver = pnp_solver.PoseEstimator((720, 1280))


def thread_func(debug_predictor, backend_server):
    while True:
        line = backend_server.stdout.readline()
        debug_predictor.kek = line
        try:
            json_dict = json.loads(line)
            debug_predictor.kek = json_dict
            if json_dict["type"] == "new_corner":
                drawer.button_callback()
            if json_dict["type"] == "gaze_vector_change":
                x = json_dict["value"]["x_right"]
                z = json_dict["value"]["z_right"]
                debug_predictor.right_eye_gaze = np.array([x, 1., z])
                x = json_dict["value"]["x_left"]
                z = json_dict["value"]["z_left"]
                debug_predictor.left_eye_gaze = np.array([x, 1., z])
            if json_dict["type"] == "matrix_change":
                matrix = json_dict["value"]
                matrix = np.array(list((map(lambda x: float(x), matrix[1: -1].split(', '))))).reshape((3, 4))
                debug_predictor.world_to_camera = matrix
                face_points = []
                for point in debug_predictor.non_transformed:
                    face_points.append(np.matmul(matrix, point))
                debug_predictor.face_points = np.array(face_points)
                debug_predictor.rotation_vector, _ = cv2.Rodrigues(matrix[:, :3])
        except json.decoder.JSONDecodeError:
            pass
        except TypeError:
            k = 3


class DebugPredictor(BasicPredictor):
    def __init__(self, noised=True):
        self.backend = subprocess.Popen(["python", "backend.py"], stdout=subprocess.PIPE)
        frontend = subprocess.Popen(["./frontend/node_modules/.bin/electron", "./frontend"])
        self.noised = noised
        self.kek = "azaz"
        self.left_eye_gaze = np.array([0, -1, 0])
        self.right_eye_gaze = np.array([0, -1, 0])
        self.plane = np.array([0., 1., 0., -100.])
        thread = threading.Thread(target=thread_func, args=(self, self.backend))
        thread.start()
        self.solver = pnp_solver.PoseEstimator((720, 1280))
        self.client = zerorpc.Client()
        self.client.connect("tcp://127.0.0.1:4242")
        self.world_to_camera = np.zeros((3, 4), dtype=np.float64)
        self.world_to_camera[0, 0] = 1
        self.world_to_camera[1, 1] = 1
        self.world_to_camera[2, 2] = 1
        self.non_transformed = np.ones((68, 4))
        self.non_transformed[:, :-1] = np.copy(self.solver.model_points_68)
        self.non_transformed.transpose()
        self.face_points = self.solver.model_points_68
        self.rotation_vector = np.array([0., 0., 0.])

    def predict_eye_vector_and_face_points(self, imgs, time_now, configurator=None):
        projections, _ = cv2.projectPoints(self.face_points, np.array([0., 0., 0.]), np.array([0., 0., 0.], ),
                                           self.solver.camera_matrix, self.solver.dist_coeefs)
        projections = projections.reshape((-1, 2))
        right_eye_gaze = np.copy(self.right_eye_gaze)
        left_eye_gaze = np.copy(self.left_eye_gaze)
        if self.noised:
            right_eye_gaze += np.random.normal(0, 0.1, (3,))
            left_eye_gaze += np.random.normal(0, 0.1, (3,))
            projections += np.random.normal(0, 2, (68, 2))
        return imgs, [right_eye_gaze], [left_eye_gaze], [projections], {}

    def get_mouse_coords(self, cameras, time_now):
        face, results, out_inform = self.predict(cameras, time_now)
        # out_inform["rotator"] -= angle
        # out_inform["cutter"] -= offset
        print(str(out_inform))

        result = list(results)
        results = [0, 0]
        for cam in result:
            for eye in cam:
                results[0] += eye[0]
                results[1] += eye[1]
        results[0] /= len(result) * 2
        results[1] /= len(result) * 2
        self.client.set_mouse_position(*results)


class GazeMLPredictor(BasicPredictor):
    def __init__(self):
        self.gazeML_process = subprocess.Popen(
            ["python", "./from_internet_or_for_from_internet/GazeML/src/elg_demo.py"],
            stdout=subprocess.PIPE)
        self.eye_one_target = np.array([0, 0, 1])
        self.eye_two_target = np.array([0, 0, 1])
        self.np_points = np.zeros((68, 2))
        def read_from_subprocess(object, process):
            while True:
                line = process.stdout.readline()
                try:
                    dict = json.loads(line)
                    if dict["eye_number"] == 0:
                        object.eye_two_target = np.array([dict["x"], -dict["y"], dict["z"]])
                    else:
                        object.eye_one_target = np.array([dict["x"], -dict["y"], dict["z"]])
                    object.np_points = np.array(dict['np_points']).reshape((68, 2))
                except json.decoder.JSONDecodeError:
                    pass

        thread_func = functools.partial(read_from_subprocess, self, self.gazeML_process)
        thread = threading.Thread(target=thread_func)
        thread.start()

    def predict_eye_vector_and_face_points(self, imgs, time_now, configurator=None):
        time.sleep(0.2)
        return [np.zeros((720, 1280))], [self.eye_one_target], [self.eye_two_target], [self.np_points], {}