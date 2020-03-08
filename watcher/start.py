import time
import tkinter as tk

import cv2
from cv2 import VideoCapture

import data_enhancer.eye_vector_enhancer
from camera_holders import camera_system_factory
import camera_holders
import data_enhancer
import from_internet_or_for_from_internet.PNP_solver as PNP_solver
import drawer

import predictor_module.visual_debug_predictor

DEBUG_PREDICTOR = True
NO_CALIB_DEBUG = False

app = drawer.App(tk.Tk(), "Tkinter and OpenCV", drawer.button_callback)
# Create a window and pass it to the Application object

cam = VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam = camera_holders.StubCameraHolder(cam)
cameras = list()
cameras.append(cam)
solver = PNP_solver.PoseEstimator((720, 1280))

if DEBUG_PREDICTOR:
    predictor_obj = predictor_module.VisualDebugPredictor()
else:
    predictor_obj = predictor_module.GazeMLPredictor()

if not NO_CALIB_DEBUG:
    drawer.cycling_flag = True
    while drawer.cycling_flag:
        for camera in cameras:
            img = camera.get_picture()
            # img = img.resize((500, 500))
            try:
                # enhancer = data_enhancer.WidthHeightDataEnhancer(text_size=30)
                # pic, output = enhancer.process(face, np_points)

                enhancer = data_enhancer.eye_vector_enhancer.EyeVectorEnhancer(draw_points=True)

                faces, eye_one_vectors, eye_two_vectors, np_points, _ = \
                    predictor_obj.predict_eye_vector_and_face_points([img], time.time())

                # pic, output = enhancer.process(faces[0], np_points[0], eye_one_vectors[0], eye_two_vectors[0])
                app.draw_image(faces[0], max_size="large")

                # pyautogui.moveTo(1920 - results[0]*10, results[1]*10)
            except:
                app.draw_image(img)

    for corner in camera_system_factory.corner_dict:
        drawer.cycling_flag = True
        app.change_corner(corner)
        last_time = time.time()
        while drawer.cycling_flag:
            time_now = time.time()
            for camera in cameras:
                try:
                    cur_time = time.time()
                    app.draw_eye(None, camera.calibration_tick(cur_time, predictor_obj))
                    last_time = cur_time
                    # pyautogui.moveTo(1920 - results[0]*10, results[1]*10)
                except:
                    pass
        for camera in cameras:
            camera.calibration_corner_end(corner)

    for camera in cameras:
        camera.calibration_end()

while True:
    time_now = time.time()
    try:
        predictor_obj.move_mouse_to_gaze_pixel(cameras, time_now)
    except Exception as e:
        e = e
    #   app.draw_image(face)
