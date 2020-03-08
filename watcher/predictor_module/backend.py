import base64
import math
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import numpy as np
import zerorpc
from PIL import Image

import from_internet_or_for_from_internet.PNP_solver as PNP_solver
from camera_holders import seen_to_screen


# import camera_holders

def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    buf = buf[:, :, 0:3]
    return Image.fromarray(buf)


def fig2data(fig: plt.Figure):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
    buf.shape = (h, w, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll(buf, 3, axis=2)
    return buf


def get_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Transform rotations around 3 axis to a rotation matrix

    :param angles: shape [3], rotations around each axis
    :return: shape [3, 3], rotation matrix
    """
    x_matrix = np.array([[1, 0, 0],
                         [0, math.cos(angles[0]), -math.sin(angles[0])],
                         [0, math.sin(angles[0]), math.cos(angles[0])]
                         ])
    y_matrix = np.array([[math.cos(angles[1]), 0, math.sin(angles[1])],
                         [0, 1, 0],
                         [-math.sin(angles[1]), 0, math.cos(angles[1])]
                         ])
    z_matrix = np.array([[math.cos(angles[2]), -math.sin(angles[2]), 0],
                         [math.sin(angles[2]), math.cos(angles[2]), 0],
                         [0, 0, 1]
                         ])
    return np.matmul(z_matrix, np.matmul(y_matrix, x_matrix))


class BackendForDebugPredictor:
    """Used for visual debug all watcher - electron debug interaction

    All visual debug predictor logic is implemented in this class

    Should be run as a subprocess as gevent doesnt like threads
    For electron -> backend interaction remote procedure call via zerorpc is used
    Backend -> electron requests are impossible in this configuration
    Backend -> watcher interaction is via stdout in json format
    Watcher -> backend via zerorpc
    All APIs are structured with comments

    :ivar _draw_full_face: (bool), whether to draw all face points in 3d scene
    :ivar _draw_corner_vectors: (bool), whether to draw vectors from eyes to corners of the screen
                                should be false if head is moved in the process
    :ivar _corner_vectors: (List[np.ndarray]), list of 3d vectors from right eye at at the start
                           it'll be wrong if head is moved
    :ivar _solver: (pnp_solver.PoseEstimator)
    :ivar _eye_right: (np.ndarray, shape [3]), current right eye position in camera coordinate system
    :ivar _eye_left: (np.ndarray, shape [3]), current left eye position in camera coordinate system
    :ivar _eye_right_origin: (np.ndarray, shape [3]), right eye position in face coordinate system
    :ivar _eye_left_origin: (np.ndarray, shape [3]), left eye position in face coordinate system
    :ivar _eyes: (np.ndarray, shape [2, 3]), list of all eyes current position, packed for comfort
    :ivar _plane: (list[float], shape [4]), screen plane ax + by + cz + d = 0 in format [a, b, c, d]
    :ivar _corner_points: (list[np.ndarray]), list of all screen corner points in 3d
                          in camera coordinate system
    :ivar _view_vector: (np.ndarray), current gaze vector in camera coordinate system
    :ivar _to_predict_status: (bool), whether to draw predicted gaze target point
    :ivar _mouse_coords: (np.ndarray, shape [2]), current predicted pixel target of gaze
    :ivar _rotation_angles: (np.ndarray, shape [3]), current head rotation angles around x, y and z axis
                            in radians
    :ivar _translation_vector: (np.ndarray, shape [3]), current head translation
    :ivar _all_points: (np.ndarray, shape [68, 3]) current facial landmarks position in camera coordinate system
    """

    def __init__(self, draw_corner_vectors=False, draw_full_face=True):
        """Constructs object

        :param draw_corner_vectors: (bool), Drawing corner vectors shouldn't be used if head is moved
        :param draw_full_face: (bool), whether to draw all face points in 3d
        """
        if draw_corner_vectors:
            warnings.warn("Drawing corner vectors shouldn't be used if head is moved")

        self._draw_full_face = draw_full_face
        self._draw_corner_vectors = draw_corner_vectors
        self._corner_vectors = []

        self._solver = PNP_solver.PoseEstimator()

        self._eye_right = sum(self._solver.model_points_68[36:41]) / 6
        self._eye_left = sum(self._solver.model_points_68[42:47]) / 6
        self._eye_right_origin = np.append(self._eye_right, [1])
        self._eye_left_origin = np.append(self._eye_left, [1])
        self._eyes = np.array([self._eye_right, self._eye_left])

        self._plane = np.array([0, 1., 0., -100.])
        self._corner_points = []
        self._view_vector = np.array([0., 1., 0.])
        self._to_predict_status = False
        self._mouse_coords = np.array([0, 0, 0])
        self._rotation_angles = np.array([0., 0., 0.])
        self._translation_vector = np.array([0., 0., 0.])
        self._all_points = np.copy(self._solver.model_points_68)

    # API electron-server-----------------------------

    def request(self, x: float, z: float) -> str:
        """Get new both eyes gaze vectors by gaze from right eye and redraw the picture

        This call also notifies watcher about gaze vectors change

        :param x: x component of gaze vector in [-100, 100]
        :param z: z component of gaze vector in [-100, 100]
        :return: a base64 encoded jpg picture of the plots
        """
        x, z = int(x) / 100, int(z) / 100

        left_eye_vector = self.get_left_eye_vector(np.array([x, 1., z]))

        # json-formatted output string for watcher
        eyes = f'{{"x_right": {x}, "z_right": {z}, "x_left": {left_eye_vector[0]}, "z_left": {left_eye_vector[2]}}}'
        json_request = f'{{"type": "gaze_vector_change", "value": {eyes}}}'
        print(json_request)

        self._view_vector = np.array([x, 1., z])
        img_str = self.get_plot_pic(x, z)
        return img_str

    def new_corner(self, x: int, z: int) -> str:
        """End current corner remembering of camera system factory

        This call notifies watcher

        :param x: x component of gaze vector in [-100, 100]
        :param z: z component of gaze vector in [-100, 100]
        :return: str
        """
        x, z = int(x), int(z)

        self._corner_vectors.append(np.array([x / 100, 1, z / 100]))
        self._corner_points.append(self._get_plane_line_cross_point(self._eye_right, self._corner_vectors[-1]))

        json_request = f'{{"type": "new_corner"}}'
        print(json_request)
        return "kek"

    def head_translation(self, x: int, y: int, z: int) -> None:
        """Change head translation in camera coordinate system

        :param x: new x position of head
        :param y: new y position of head
        :param z: new z position of head
        :return:
        """
        self._translation_vector = np.array([x, y, z], dtype=np.float64)
        self._recalculate_face_position()

    def head_rotation(self, x: int, y: int, z: int):
        """Change head rotation in camera coordinate system

        :param x: new rotation arount x axis, in degree
        :param y: new rotation arount y axis, in degree
        :param z: new rotation arount z axis, in degree
        :return:
        """
        self._rotation_angles = np.array([x, y, z], dtype=np.float64) / 180 * math.pi
        self._rotation_angles[0] -= math.pi / 2

        self._recalculate_face_position()

    # API watcher-server -----------

    def set_mouse_position(self, x: int, y: int):
        """Set on screen pixel cursor position

        :param x:
        :param y:
        :return:
        """
        self._mouse_coords = np.array([x, y])
        self._to_predict_status = True

    # Non-api ----------------------------

    def get_plot_pic(self, x: float, z: float) -> str:
        """Draw current visual picture of system

        :param x: [-1, 1]
        :param z: [-1, 1]
        :return: base64 encoded jpg picture
        """
        plt.close('all')
        fig = plt.figure()

        # 3d subplot ----------

        ax = fig.add_subplot(2, 2, 1, projection='3d')

        ax.scatter(self._eyes[:, 0], self._eyes[:, 1], self._eyes[:, 2])

        self._draw_plane(ax)

        if self._draw_corner_vectors:
            for vect in self._corner_vectors:
                self._draw_vector_from_right_eye(ax, vect, point_needed=True)

        self._draw_vector_from_right_eye(ax, np.array([x, 1, z]), color="#00ff00", point_needed=True)

        arr = np.array(self._corner_points).reshape((-1, 3))

        # draw face
        if self._draw_full_face:
            ax.scatter(self._all_points[:, 0], self._all_points[:, 1], self._all_points[:, 2])

        # draw screen borders
        if arr.shape[0] >= 5:
            arr[4] = arr[0]
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="#ff0000")

        # 2d projections to the screen plane -------

        ax = fig.add_subplot(2, 2, 2)

        # draw corner points
        arr = np.array(self._corner_points).reshape((-1, 3))
        ax.scatter(arr[:, 0], arr[:, 2], color="#880000")

        # draw view exact point
        view = self._get_plane_line_cross_point(self._eye_right, np.array([x, 1., z]))
        ax.scatter(view[0], view[2], color="#00ff00")

        # draw screen border
        arr = np.array(self._corner_points).reshape((-1, 3))
        if arr.shape[0] >= 5:
            arr[4] = arr[0]
            ax.plot(arr[:, 0], arr[:, 2], color="#ff0000")

        # 2d on rectangular screen with size 1920 x 1080 ------------

        ax = fig.add_subplot(2, 2, 3)

        # draw corners of the screen
        points = np.array(seen_to_screen.points_in_square) * [1920, 1080]
        points = points.transpose()
        ax.scatter(*points)

        # draw screen borders
        points[:, -1] = points[:, 0]
        ax.plot(*points)

        # draw cursor position
        if self._to_predict_status:
            ax.scatter(*self._mouse_coords)
        # plt to img

        image = fig2img(fig)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str

    def _recalculate_face_position(self) -> None:
        """Recalculates current face landmarks position according to rotation and translation

        Also notifies watcher about change of face position
        Should be called each time face changes its position

        :return:
        """
        matrix = np.zeros((3, 4))
        matrix[:, :3] = get_rotation_matrix(self._rotation_angles)
        matrix[:, 3] = self._translation_vector

        if self._draw_full_face:
            for i in range(68):
                self._all_points[i] = np.matmul(matrix, [*(self._solver.model_points_68[i]), 1])

        self._eye_left = np.matmul(matrix, self._eye_left_origin)
        self._eye_right = np.matmul(matrix, self._eye_right_origin)
        self._eyes = np.array([self._eye_left, self._eye_right])

        string = matrix.reshape((-1,))
        string = list(string)
        json_request = f'{{"type": "matrix_change", "value": "{string}"}}'
        print(json_request)

    def _draw_vector_from_right_eye(self, plt_axis: plt.Axes, vect: np.ndarray,
                                    color: str = "#880000", point_needed: bool = False) -> None:
        """Draws vectors from right and left eye to the cross point of right eye vector and plane

        :param plt_axis: axes where to draw
        :param vect: right eye gaze vector
        :param color: a "#XXXXXX" encoded color string
        :param point_needed: whether big point at the cross is needed
        :return:
        """
        vect = np.array(vect)

        point = self._get_plane_line_cross_point(self._eye_right, vect)
        tmp = [self._eye_left, point, self._eye_right]
        tmp = np.array(tmp)

        plt_axis.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2], color)

        if point_needed:
            plt_axis.scatter(*point, color=color)

    def _draw_plane(self, axis: Axes3D) -> None:
        """Draw plane that is not perpendicular to z and x axes

        :param axis: where to draw
        :return:
        """
        xx, zz = np.meshgrid(range(-150, 151, 50), range(-150, 151, 50))
        a, b, c, d = self._plane
        yy = - (d + a * xx + c * zz) / b
        axis.plot_wireframe(xx, yy, zz)

    def _get_plane_line_cross_point(self, point: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Get cross point of plane and line(point and vector)

        :param point: one point of line
        :param vector: line vector
        :return: shape [3], cross point
        """
        t = - np.matmul([*point, 1], self._plane) / np.matmul(vector, self._plane[0:3])
        return point + (vector * t)

    def get_left_eye_vector(self, right_eye_vector: np.ndarray) -> np.ndarray:
        """Get normalized gaze vector from left

        It is implied that second gaze is to the cross point of line and screen plane

        :param right_eye_vector: shape [3], right eye gaze vector
        :return: shape [3], left eye gaze vector
        """
        point = self._get_plane_line_cross_point(self._eye_right, right_eye_vector)

        res = point - self._eye_left
        res /= res[1]
        return res


if __name__ == "__main__":
    backend = BackendForDebugPredictor()
    server = zerorpc.Server(backend)
    server.bind("tcp://0.0.0.0:4242")
    server.run()
