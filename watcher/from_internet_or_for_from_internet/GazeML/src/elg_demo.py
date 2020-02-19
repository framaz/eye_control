#!/usr/bin/env python3/home/framaz/eye_control/watcher/drawer.py
"""Main script for gaze direction inference from webcam feed."""
import argparse
import copy
import math
import os
import queue
import threading
import time
import json
import coloredlogs
import cv2
import cv2 as cv
import dlib
import numpy as np
import tensorflow as tf
from PIL import Image
from datasources import Video, Webcam
from models import ELG
import util.gaze

faces_folder_path = "faces"
predictor_path = "68_predictor.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

class BasicTranslation:
    def __init__(self, np_points):
        self.np_points_before = copy.deepcopy(np_points)

    def apply_forth(self, face):
        raise NotImplementedError
        pass

    def apply_back(self):
        raise NotImplementedError
        pass

    def __str__(self):
        return type(self).__name__

    def get_modification_data(self):
        raise NotImplementedError
PADDING = 2

class FaceCropper(BasicTranslation):
    def __init__(self, np_points, face=None):
        # TODO save width/height of pic before cropping
        super().__init__(np_points)
        self.face = face
        self.left = min(np_points[:, 0]) - PADDING
        self.right = max(np_points[:, 0]) + PADDING
        self.top = min(np_points[:, 1]) - PADDING
        self.bottom = max(np_points[:, 1]) + PADDING

        offset = np.array([min(np_points[:, 0]) - PADDING, min(np_points[:, 1]) - PADDING],
                          dtype=np.float32)
        for i in range(np_points.shape[0]):
            np_points[i] -= offset
        self.offset = offset
        self.result_offset = None
        self.np_points_after = copy.deepcopy(np_points)
        self.width = 0
        self.heigth = 0
        self.resize_ratio_x = 0
        self.resize_ratio_y = 0

    def apply_forth(self, face):
        if not (isinstance(face, np.ndarray)):
            face = np.array(face)
        np_points = copy.deepcopy(self.np_points_before)
        if not (self.face is None):
            [self.width, self.heigth] = self.face.shape
            self.resize_ratio_x = face.shape[0] / self.width
            self.resize_ratio_y = face.shape[1] / self.heigth
            np_points *= [self.resize_ratio_x, self.resize_ratio_y]
        if type(face) is np.ndarray:
            face = Image.fromarray(face)
        face = face.crop((min(np_points[:, 0]) - PADDING, min(np_points[:, 1]) - PADDING,
                          max(np_points[:, 0]) + PADDING, max(np_points[:, 1]) + PADDING,))
        self.result_offset = min(np_points[:, 0]) - PADDING, min(np_points[:, 1]) - PADDING
        return face, copy.deepcopy(self.np_points_after)

    def __str__(self):
        return f"{super().__str__()}: {self.offset}"

    def get_modification_data(self):
        return self.offset,

    def get_resulting_offset(self):
        return self.result_offset


def detect_face_points_dlib(src, cropping_needed=True):
    img = copy.deepcopy(src)
    img = np.flip(img, axis=1)
    img_out = copy.deepcopy(img)
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
        resizing_cropper = FaceCropper(np_points, smaller_img)
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
            return np_points, resizing_cropper, img_out
    except:
        pass


if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Demonstration of landmarks localization.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--headless', action='store_true')

    parser.add_argument('--fps', type=int, default=60, help='Desired sampling rate of webcam')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of webcam to use')

    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Check if GPU is available
    from tensorflow.python.client import device_lib
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices()
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except:
        pass

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:

        # Declare some parameters
        batch_size = 2

        # Define webcam stream data source
        # Change data_format='NHWC' if not using CUDA
        if args.from_video:
            assert os.path.isfile(args.from_video)
            data_source = Video(args.from_video,
                                tensorflow_session=session, batch_size=batch_size,
                                data_format='NCHW' if gpu_available else 'NHWC',
                                eye_image_shape=(108, 180))
        else:
            data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
                                 camera_id=args.camera_id, fps=args.fps,
                                 data_format='NCHW' if gpu_available else 'NHWC',
                                 eye_image_shape=(36, 60))

        # Define model
        if args.from_video:
            model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=3,
                num_modules=3,
                num_feature_maps=64,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )
        else:
            model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=1,
                num_modules=2,
                num_feature_maps=32,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )

        # Record output frames to file if requested
        if args.record_video:
            video_out = None
            video_out_queue = queue.Queue()
            video_out_should_stop = False
            video_out_done = threading.Condition()

            def _record_frame():
                global video_out
                last_frame_time = None
                out_fps = 30
                out_frame_interval = 1.0 / out_fps
                while not video_out_should_stop:
                    frame_index = video_out_queue.get()
                    if frame_index is None:
                        break
                    assert frame_index in data_source._frames
                    frame = data_source._frames[frame_index]['bgr']
                    h, w, _ = frame.shape
                    if video_out is None:
                        video_out = cv.VideoWriter(
                            args.record_video, cv.VideoWriter_fourcc(*'H264'),
                            out_fps, (w, h),
                        )
                    now_time = time.time()
                    if last_frame_time is not None:
                        time_diff = now_time - last_frame_time
                        while time_diff > 0.0:
                            video_out.write(frame)
                            time_diff -= out_frame_interval
                    last_frame_time = now_time
                video_out.release()
                with video_out_done:
                    video_out_done.notify_all()
            record_thread = threading.Thread(target=_record_frame, name='record')
            record_thread.daemon = True
            record_thread.start()

        # Begin visualization thread
        inferred_stuff_queue = queue.Queue()

        def _visualize_output():
            last_frame_index = 0
            last_frame_time = time.time()
            fps_history = []
            all_gaze_histories = []

            while True:
                # If no output to visualize, show unannotated frame
                if inferred_stuff_queue.empty():
                    continue

                # Get output from neural network and visualize
                output = inferred_stuff_queue.get()
                bgr = None
                for j in range(batch_size):
                    frame_index = output['frame_index'][j]
                    if frame_index not in data_source._frames:
                        continue
                    frame = data_source._frames[frame_index]
                    # Decide which landmarks are usable
                    heatmaps_amax = np.amax(output['heatmaps'][j, :].reshape(-1, 18), axis=0)
                    can_use_eye = np.all(heatmaps_amax > 0.7)
                    can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
                    can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)

                    start_time = time.time()
                    eye_index = output['eye_index'][j]
                    bgr = frame['bgr']
                    eye = frame['eyes'][eye_index]
                    eye_image = eye['image']
                    eye_side = eye['side']
                    eye_landmarks = output['landmarks'][j, :]
                    eye_radius = output['radius'][j][0]
                    if eye_side == 'left':
                        eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                        eye_image = np.fliplr(eye_image)

                    # Embed eye image and annotate for picture-in-picture
                    eye_upscale = 2
                    eye_image_raw = cv.cvtColor(cv.equalizeHist(eye_image), cv.COLOR_GRAY2BGR)
                    eye_image_raw = cv.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
                    eye_image_annotated = np.copy(eye_image_raw)
                    face_index = int(eye_index / 2)
                    eh, ew, _ = eye_image_raw.shape
                    v0 = face_index * 2 * eh
                    v1 = v0 + eh
                    v2 = v1 + eh
                    u0 = 0 if eye_side == 'left' else ew
                    u1 = u0 + ew
                    bgr[v0:v1, u0:u1] = eye_image_raw
                    bgr[v1:v2, u0:u1] = eye_image_annotated

                    # Visualize preprocessing results
                    frame_landmarks = (frame['smoothed_landmarks']
                                       if 'smoothed_landmarks' in frame
                                       else frame['landmarks'])

                    # Transform predictions
                    eye_landmarks = np.concatenate([eye_landmarks,
                                                    [[eye_landmarks[-1, 0] + eye_radius,
                                                      eye_landmarks[-1, 1]]]])
                    eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                                       'constant', constant_values=1.0))
                    eye_landmarks = (eye_landmarks *
                                     eye['inv_landmarks_transform_mat'].T)[:, :2]
                    eye_landmarks = np.asarray(eye_landmarks)
                    eyelid_landmarks = eye_landmarks[0:8, :]
                    iris_landmarks = eye_landmarks[8:16, :]
                    iris_centre = eye_landmarks[16, :]
                    eyeball_centre = eye_landmarks[17, :]
                    eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                                    eye_landmarks[17, :])

                    # Smooth and visualize gaze direction
                    num_total_eyes_in_frame = len(frame['eyes'])
                    if len(all_gaze_histories) != num_total_eyes_in_frame:
                        all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
                    gaze_history = all_gaze_histories[eye_index]
                    if can_use_eye:
                        # Visualize landmarks
                        # cv.circle(  # Eyeball outline
                        #     bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                        #     int(np.round(eyeball_radius)), color=(0, 255, 0),
                        #     thickness=1, lineType=cv.LINE_AA,
                        # )

                        # Draw "gaze"
                        # from models.elg import estimate_gaze_from_landmarks
                        # current_gaze = estimate_gaze_from_landmarks(
                        #     iris_landmarks, iris_centre, eyeball_centre, eyeball_radius)
                        i_x0, i_y0 = iris_centre
                        e_x0, e_y0 = eyeball_centre
                        theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
                        phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                                -1.0, 1.0))
                        current_gaze = np.array([theta, phi])
                        gaze_history.append(current_gaze)
                        gaze_history_max_len = 10
                        if len(gaze_history) > gaze_history_max_len:
                            gaze_history = gaze_history[-gaze_history_max_len:]
                        [theta, phi] = np.mean(gaze_history, axis=0)
                        y = -math.sin(theta)
                        z = math.sqrt((math.cos(theta)**2 / (1+(math.tan(phi))**2)))
                        x = z*math.tan(phi)
                        np_points, _, _ = detect_face_points_dlib(frame['grey'], cropping_needed=False)
                        json_string = f'{{"eye_number": {j}, "x": {x}, "y": {y}, "z": {z},' \
                                      f' "np_points": {list(np_points.reshape((-1, )))} }}'
                        print(json_string)
                    else:
                        gaze_history.clear()

       # visualize_thread = threading.Thread(target=_visualize_output, name='visualization')
       # visualize_thread.daemon = True
       # visualize_thread.start()

        # Do inference forever
        infer = model.inference_generator()
        all_gaze_histories = []
        frame_gen = data_source.frame_generator()
        def _dlib_landmark_gen(camera_frame_gen):
            i = 7
            last_np_points = np.zeros((68, 2))
            while True:
                if i == 7:
                    img = next(camera_frame_gen)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    try:
                        last_np_points, _, _ = detect_face_points_dlib(img, cropping_needed=False)
                        i = 0
                        yield last_np_points
                    except Exception as esx:
                        aasd=3
                else:
                    i += 1
                    yield last_np_points
        np_points_gen = _dlib_landmark_gen(frame_gen)
        cur_time = time.time()
        time_history = []
        while True:
            output = next(infer)
            time_history.append(time.time() - cur_time)
            cur_time = time.time()
            for frame_index in np.unique(output['frame_index']):
                if frame_index not in data_source._frames:
                    continue
                frame = data_source._frames[frame_index]
                if 'inference' in frame['time']:
                    frame['time']['inference'] += output['inference_time']
                else:
                    frame['time']['inference'] = output['inference_time']
            bgr = None
            for j in range(batch_size):
                frame_index = output['frame_index'][j]
                if frame_index not in data_source._frames:
                    continue
                frame = data_source._frames[frame_index]
                # Decide which landmarks are usable
                heatmaps_amax = np.amax(output['heatmaps'][j, :].reshape(-1, 18), axis=0)
                can_use_eye = np.all(heatmaps_amax > 0.7)
                can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
                can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)

                start_time = time.time()
                eye_index = output['eye_index'][j]
                bgr = frame['bgr']
                eye = frame['eyes'][eye_index]
                eye_image = eye['image']
                eye_side = eye['side']
                eye_landmarks = output['landmarks'][j, :]
                eye_radius = output['radius'][j][0]
                if eye_side == 'left':
                    eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                    eye_image = np.fliplr(eye_image)

                # Embed eye image and annotate for picture-in-picture
                eye_upscale = 2
                eye_image_raw = cv.cvtColor(cv.equalizeHist(eye_image), cv.COLOR_GRAY2BGR)
                eye_image_raw = cv.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
                eye_image_annotated = np.copy(eye_image_raw)
                face_index = int(eye_index / 2)
                eh, ew, _ = eye_image_raw.shape
                v0 = face_index * 2 * eh
                v1 = v0 + eh
                v2 = v1 + eh
                u0 = 0 if eye_side == 'left' else ew
                u1 = u0 + ew
                bgr[v0:v1, u0:u1] = eye_image_raw
                bgr[v1:v2, u0:u1] = eye_image_annotated

                # Visualize preprocessing results
                frame_landmarks = (frame['smoothed_landmarks']
                                   if 'smoothed_landmarks' in frame
                                   else frame['landmarks'])

                # Transform predictions
                eye_landmarks = np.concatenate([eye_landmarks,
                                                [[eye_landmarks[-1, 0] + eye_radius,
                                                  eye_landmarks[-1, 1]]]])
                eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                                   'constant', constant_values=1.0))
                eye_landmarks = (eye_landmarks *
                                 eye['inv_landmarks_transform_mat'].T)[:, :2]
                eye_landmarks = np.asarray(eye_landmarks)
                eyelid_landmarks = eye_landmarks[0:8, :]
                iris_landmarks = eye_landmarks[8:16, :]
                iris_centre = eye_landmarks[16, :]
                eyeball_centre = eye_landmarks[17, :]
                eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                                eye_landmarks[17, :])

                # Smooth and visualize gaze direction
                num_total_eyes_in_frame = len(frame['eyes'])
                if len(all_gaze_histories) != num_total_eyes_in_frame:
                    all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
                gaze_history = all_gaze_histories[eye_index]
                if can_use_eye:
                    # Visualize landmarks
                    # cv.circle(  # Eyeball outline
                    #     bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                    #     int(np.round(eyeball_radius)), color=(0, 255, 0),
                    #     thickness=1, lineType=cv.LINE_AA,
                    # )

                    # Draw "gaze"
                    # from models.elg import estimate_gaze_from_landmarks
                    # current_gaze = estimate_gaze_from_landmarks(
                    #     iris_landmarks, iris_centre, eyeball_centre, eyeball_radius)
                    i_x0, i_y0 = iris_centre
                    e_x0, e_y0 = eyeball_centre
                    theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
                    phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                            -1.0, 1.0))
                    current_gaze = np.array([theta, phi])
                    gaze_history.append(current_gaze)
                    gaze_history_max_len = 5
                    if len(gaze_history) > gaze_history_max_len:
                        gaze_history = gaze_history[-gaze_history_max_len:]
                    [theta, phi] = np.mean(gaze_history, axis=0)
                    y = -math.sin(theta)
                    z = math.sqrt((math.cos(theta) ** 2 / (1 + (math.tan(phi)) ** 2)))
                    x = z * math.tan(phi)
                    try:
                        np_points = next(np_points_gen)
                        json_string = f'{{"eye_number": {j}, "x": {x}, "y": {y}, "z": {z},' \
                                      f' "np_points": {list(np_points.reshape((-1,)))} }}'
                        print(json_string)
                    except TypeError as err:
                        kek = 9
                else:
                    gaze_history.clear()

            inferred_stuff_queue.put_nowait(output)

          #  if not visualize_thread.isAlive():
           #     break

            if not data_source._open:
                break

        # Close video recording
        if args.record_video and video_out is not None:
            video_out_should_stop = True
            video_out_queue.put_nowait(None)
            with video_out_done:
                video_out_done.wait()
