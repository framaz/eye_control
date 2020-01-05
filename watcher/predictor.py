from functools import partial
import collections
import dlib
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import time
import calibrator
import eye_module
import copy
import model as model_gen
import matplotlib.pyplot as plt
def process_pic(src):
    img = src
    cur_time = time.time()
    smaller_img = copy.deepcopy(img)
    smaller_img = Image.fromarray(smaller_img)
    smaller_img = smaller_img.resize((512, int(512 * smaller_img._size[1]/smaller_img._size[0])))
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

        return np_points, resizing_cropper, img
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
    k = d / (tf.math.reduce_sum(tens * fl, 1) + kek)
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


def calibration_predict(img, time_now):
    _, point = predict(img, time_now)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def predict(img, time_now, configurator=None):
    np_points, resizing_cropper, img = process_pic(np.array(img))
    if np_points is not None:
        try:
            tmp_out_inform = {}
            #    np_points[i] = rotate_point(np_points[i], middle, -angle)
            face_cutter = calibrator.FaceCropper(np_points)
            face, np_points = face_cutter.apply_forth(img)
            rotator = calibrator.RotationTranslation(np_points)
            face, np_points = rotator.apply_forth(face)
            tmp_out_inform["cutter"] = face_cutter.get_modification_data()
            tmp_out_inform["rotator"] = rotator.get_modification_data()
            eyes = []
            eyes.append(eye_module.process_eye(face, np_points[36:42]))
            eyes.append(eye_module.process_eye(face, np_points[42: 48]))
            eyes_to_predict = []
            for eye, _, _ in eyes:
                eyes_to_predict.append(eye)
            res = model.predict(np.array(eyes_to_predict))
            v1 = normalize(res[0])
            v2 = normalize(res[1])
            print(v1)
            res_pixel = pixel_func(res)
            res_pixel = res_pixel.numpy()
            for i in range(res_pixel.shape[0]):
                history.append((res_pixel[i, 0], res_pixel[i, 1], time_now))
            results = calibrator.smooth_n_cut(history, time_now)
            if not isinstance(face, Image.Image):
                face = Image.fromarray(face)
            drawer = ImageDraw.Draw(face)
            #   for [x, y] in np_points:
            #   #     drawer.ellipse([x-2, y-2, x+2, y+2], fill=0)
            eye_middle = np.average(np_points[36:41], axis=0)
            drawer.ellipse([eye_middle[0] - 2, eye_middle[1] - 2, eye_middle[0] + 2, eye_middle[1] + 2])
            face = face.resize((500, 500))
            face = eyes_to_predict[0] * 255
            face = face.astype(dtype=np.uint8).reshape(36, 60)
            face = Image.fromarray(face)
            face = face.resize((60 * 5, 36 * 5))
            return face, results, tmp_out_inform
        except:
            pass
