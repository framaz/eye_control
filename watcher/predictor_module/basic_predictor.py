import pyautogui
from PIL import Image

import utilities


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
        faces = utilities.pack_to_one_image(*faces)
        return faces, results, tmp_out_informs