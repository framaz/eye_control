[33mcommit ca0d790f06637c475a948d3f116372c49bfde444[m[33m ([m[1;35mrefs/stash[m[33m)[m
Merge: 88b9094 81a8701
Author: framaz <framaz@yandex.ru>
Date:   Thu Feb 6 17:24:52 2020 +0500

    WIP on head_rotation_track: 88b9094 head rotation debug

[1mdiff --cc watcher/predictor_module.py[m
[1mindex abd1f78,abd1f78..ed0a16c[m
[1m--- a/watcher/predictor_module.py[m
[1m+++ b/watcher/predictor_module.py[m
[36m@@@ -4,6 -4,6 +4,7 @@@[m [mimport tim[m
  from functools import partial[m
  from math import sqrt[m
  [m
[32m++import cv2[m
  import dlib[m
  import numpy as np[m
  import pyautogui[m
[36m@@@ -115,6 -115,6 +116,23 @@@[m [mclass GoodPredictor(BasicPredictor)[m
                      face = Image.fromarray(face)[m
                      face = face.resize((60 * 5, 36 * 5))[m
                      faces.append(to_out_face)[m
[32m++[m
[32m++                    faceModel = solver.model_points_68[((36, 39, 42, 45, 48, 54),)][m
[32m++                    headpose_hr, headpose_ht = solver.solve_pose(np_points)[m
[32m++                    hR, _ = cv2.Rodrigues(headpose_hr)[m
[32m++                    matrix = np.zeros((3, 4))[m
[32m++                    matrix[:, :3] = hR[m
[32m++                    matrix[:, 3] = headpose_ht.reshape((-1, ))[m
[32m++                    Fc = [][m
[32m++                    for i in range(6):[m
[32m++                        Fc.append(np.matmul(matrix, [*(faceModel[i]), 1]))[m
[32m++                    Fc = np.array(Fc)[m
[32m++                    right_eye_center = 0.5 * (Fc[0] + Fc[1])[m
[32m++                    left_eye_center = 0.5 * (Fc[2] + Fc[3])[m
[32m++                    eye_image_width = 36[m
[32m++                    eye_image_height = 60[m
[32m++                    eye_one_vector = denormalize(eye_one_vector, right_eye_center, hR)[m
[32m++                    eye_two_vector = denormalize(eye_two_vector, left_eye_center, hR)[m
                      eye_one_vectors.append(eye_one_vector)[m
                      eye_two_vectors.append(eye_two_vector)[m
                      np_points_all.append(np_points)[m
[36m@@@ -123,6 -123,6 +141,26 @@@[m
                      pass[m
          return faces, eye_one_vectors, eye_two_vectors, np_points_all, tmp_out_informs[m
  [m
[32m++def denormalize(gvnew, target_3D, hR, distance_new=600):[m
[32m++    distance = np.sqrt(np.matmul(target_3D, target_3D))[m
[32m++    #np.seterr(divide='ignore', invalid='ignore')Z[m
[32m++    z_scale = distance_new/distance[m
[32m++    scaleMat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]][m
[32m++    hRx = hR[:,0][m
[32m++    forward = (target_3D/distance)[m
[32m++    down = np.cross(forward, hRx)[m
[32m++    down = down / norm(down)[m
[32m++    right = np.cross(down, forward)[m
[32m++    right = right / norm(right)[m
[32m++    rotMat = np.array([right, down, forward])[m
[32m++    cnvMat = scaleMat @ rotMat[m
[32m++    inv_cnv_mat = np.linalg.inv(cnvMat)[m
[32m++    vect = inv_cnv_mat @ gvnew[m
[32m++[m
[32m++    return vect/norm(vect)[m
[32m++[m
[32m++def norm(target_3D):[m
[32m++    return np.sqrt(np.matmul(target_3D, target_3D))[m
  [m
  def detect_face_points_dlib(src, cropping_needed=True):[m
      img = copy.deepcopy(src)[m
