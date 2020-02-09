import time
import tkinter as tk

import cv2
import pyautogui
from cv2 import VideoCapture

import calibrator
import camera_holders
import data_enhancer
import predictor_module
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import drawer

DEBUG_PREDICTOR = False
NO_CALIB_DEBUG = False


app = drawer.App(tk.Tk(), "Tkinter and OpenCV", drawer.button_callback)
# Create a window and pass it to the Application object

cam = VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam = camera_holders.CameraHolder(cam)
cameras = list()
cameras.append(cam)
solver = pnp_solver.PoseEstimator((1080, 1920))

if DEBUG_PREDICTOR:
    predictor_obj = predictor_module.DebugPredictor()
else:
    predictor_obj = predictor_module.GoodPredictor()


if not NO_CALIB_DEBUG:
    drawer.cycling_flag = True
    while drawer.cycling_flag:
        for camera in cameras:
            img = camera.get_picture()
            # img = img.resize((500, 500))
            try:
                enhancer = data_enhancer.WidthHeightDataEnhancer(text_size=30)
                # pic, output = enhancer.process(face, np_points)
                # enhancer = data_enhancer.HeadPositionAxisDataEnhancer()
                # pic, output = enhancer.process(face, np_points)

                enhancer = data_enhancer.HeadNEyeDataEnhancer(draw_points=True)

                faces, eye_one_vectors, eye_two_vectors, np_points, _ = predictor_obj.predict_eye_vector_and_face_points([img], time.time())

                pic, output = enhancer.process(faces[0], np_points[0], eye_one_vectors[0], eye_two_vectors[0])
                app.draw_image(pic, max_size="large")

                # pyautogui.moveTo(1920 - results[0]*10, results[1]*10)
            except:
                app.draw_image(img)
    offset, angle = 0, 0
    offset = 0
    angle = 0

    for corner in calibrator.corner_dict:
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
        predictor_obj.get_mouse_coords(cameras, time_now)
    except Exception as e:
        e = e
    #   app.draw_image(face)
