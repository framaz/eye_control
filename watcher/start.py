import asyncio
import time
import tkinter as tk
import os
import dlib
import numpy as np
from PIL import ImageTk, Image, ImageDraw
from cv2 import VideoCapture
import cv2
import calibrator
import pyautogui
import drawer
import eye_module
import predictor_module
import operator
import camera_holders
import data_enhancer
from functools import partial
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
from predictor_module import pixel_func, detect_face_points_dlib, model
from calibrator import rotate, smooth_n_cut

NO_CALIB_DEBUG = False
cycling_flag = True


def button_callback():
    global cycling_flag
    cycling_flag = False
    print("azazaza")


app = drawer.App(tk.Tk(), "Tkinter and OpenCV", button_callback)
# Create a window and pass it to the Application object

cam = VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam = camera_holders.CameraHolder(cam)
cameras = list()
cameras.append(cam)
solver = pnp_solver.PoseEstimator((1080, 1920))
predict_obj = predictor_module.GoodPredictor()
if not NO_CALIB_DEBUG:
    cycling_flag = True
    while cycling_flag:
        for camera in cameras:
            img = camera.get_picture()
            # img = img.resize((500, 500))
            try:
                enhancer = data_enhancer.WidthHeightDataEnhancer(text_size=30)
                # pic, output = enhancer.process(face, np_points)
                # enhancer = data_enhancer.HeadPositionAxisDataEnhancer()
                # pic, output = enhancer.process(face, np_points)

                enhancer = data_enhancer.HeadNEyeDataEnhancer()

                [faces], [eye_one_vectors], [eye_two_vectors], [
                    np_points], _ = predict_obj.predict_eye_vector_and_face_points([img], time.time(), )

                pic, output = enhancer.process(faces, np_points, eye_one_vectors, eye_two_vectors)
                app.draw_image(pic, max_size="large")

                # pyautogui.moveTo(1920 - results[0]*10, results[1]*10)
            except:
                app.draw_image(img)
    offset, angle = 0, 0
    offset = 0
    angle = 0

    for corner in calibrator.corner_dict:
        cycling_flag = True
        app.change_corner(corner)
        last_time = time.time()
        while cycling_flag:
            time_now = time.time()
            for camera in cameras:
                try:
                    cur_time = time.time()
                    app.draw_eye(None, camera.calibration_tick(cur_time,  predict_obj))
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
        face, results, out_inform = predict_obj.predict(cameras, time_now)

        app.draw_image(face)

        if not NO_CALIB_DEBUG:
            # out_inform["rotator"] -= angle
            # out_inform["cutter"] -= offset
            print(str(out_inform))
            app.draw_eye(None, str(results))
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
        else:
            app.draw_eye(None, str(results))
    except Exception as e:
        e = e
    #   app.draw_image(face)
