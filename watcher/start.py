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
import predictor
import operator
import camera_holders
import data_enhancer
from functools import partial
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
from predictor import pixel_func, detect_face_points_dlib, model
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
if not NO_CALIB_DEBUG:
    cycling_flag = True
    while cycling_flag:
        for camera in cameras:
            img = camera.get_picture()
            # img = img.resize((500, 500))
            try:
                np_points, _, face = predictor.detect_face_points_dlib(img, False)
                enhancer = data_enhancer.WidthHeightDataEnhancer(text_size=30)
                pic, output = enhancer.process(face, np_points)
                cur_time = time.time()
                r1, r2 = solver.solve_pose(solver.get_pose_marks(np_points))
                print(time.time()-cur_time)
                pic = solver.draw_axes(pic, r1, r2)
                pic = pic.astype(dtype=np.uint8)
                pic = Image.fromarray(pic)
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
                    app.draw_eye(None, camera.calibration_tick(cur_time))
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
        face, results, out_inform = predictor.predict(cameras, time_now)

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
