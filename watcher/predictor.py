from functools import partial
import collections
from math import sqrt

import dlib
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import time
import calibrator
import eye_module
import copy
import model as model_gen
import image_translation
import camera_holders
import matplotlib.pyplot as plt
import from_internet_or_for_from_internet.PNP_solver as pnp_solver


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


solver = pnp_solver.PoseEstimator((1080, 1920))


def predict_eye_vector_and_face_points(imgs, time_now, configurator=None):
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

                #face_cutter = calibrator.FaceCropper(np_points)
                #face, np_points = face_cutter.apply_forth(img)
                rotator = calibrator.RotationTranslation(np_points)
                face, np_points = rotator.apply_forth(img)
                to_out_face = face
                #tmp_out_inform["cutter"] = face_cutter.get_modification_data()
                #tmp_out_inform["rotator"] = rotator.get_modification_data()
                eyes = []

                eyes.append(eye_module.process_eye(face, np_points[36:42]))
                eyes.append(eye_module.process_eye(face, np_points[42:48]))
                eyes_to_predict = []
                for eye, _, _ in eyes:
                    eyes_to_predict.append(eye)
                res = model.predict(np.array(eyes_to_predict))
                eye_one_vector = normalize(res[0])
                eye_two_vector = normalize(res[1])
                face = Image.fromarray(face)
                drawer = ImageDraw.Draw(face)
                # #   for [x, y] in np_points:
                # #   #     drawer.ellipse([x-2, y-2, x+2, y+2], fill=0)
                eye_middle = np.average(np_points[36:41], axis=0)
                drawer.ellipse([eye_middle[0] - 2, eye_middle[1] - 2, eye_middle[0] + 2, eye_middle[1] + 2])
                face = face.resize((500, 500))
                face = eyes_to_predict[0] * 255
                face = face.astype(dtype=np.uint8).reshape(36, 60)
                face = Image.fromarray(face)
                face = face.resize((60 * 5, 36 * 5))
                faces.append(to_out_face)
                eye_one_vectors.append(eye_one_vector)
                eye_two_vectors.append(eye_two_vector)
                np_points_all.append(np_points)
                tmp_out_informs.append(tmp_out_inform)
            except:
                pass

    return faces, eye_one_vectors, eye_two_vectors, np_points_all, tmp_out_informs


def predict(cameras, time_now, ):
    imgs = []
    for camera in cameras:
        imgs.append(camera.get_picture())

    faces, eye_one_vectors, eye_two_vectors, np_points_all, tmp_out_informs = predict_eye_vector_and_face_points(imgs, time_now, )

    results = []
    for i, camera in zip(range(len(cameras)), cameras):
        results.append(camera.update_gazes_history(eye_one_vectors[i], eye_two_vectors[i], np_points_all[i], time_now))

    def face_to_img(face):
        if not isinstance(face, Image.Image):
            face = Image.fromarray(face)
        return face

    faces = list(map(lambda face: face_to_img(face), faces))
    faces = image_translation.pack_to_one_image(*faces)
    return faces, results, tmp_out_informs
