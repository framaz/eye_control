import zerorpc
import numpy
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import from_internet_or_for_from_internet.PNP_solver as pnp_solver


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
    w, h, d = buf.shape
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


class BackendForDebugPredictor:
    def __init__(self, draw_corner_vectors=True):
        self.draw_corner_vectors = draw_corner_vectors
        self.corner_vectors = []
        self.solver = pnp_solver.PoseEstimator()
        self.eye_right = sum(self.solver.model_points_68[36:41]) / 6
        self.eye_left = sum(self.solver.model_points_68[42:47]) / 6
        self.eye_right[:1] *= -1
        self.eye_left[:1] *= -1
        self.eyes = np.array([self.eye_right, self.eye_left])
        self.plane = np.array([0., 1., 0., -100.])
        self.corner_points = []

    # API -----------------------------

    def request(self, x, z):
        x, z = int(x) / 100, int(z) / 100
        print(x)
        fig = plt.figure(figsize=plt.figaspect(2.))

        # 3d part

        ax = fig.add_subplot(2, 1, 1, projection='3d')

        ax.scatter(self.eyes[:, 0], self.eyes[:, 1],
                   self.eyes[:, 2])
        self._draw_plane(ax)

        if self.draw_corner_vectors:
            for vect in self.corner_vectors:
                self._draw_vector_from_right_eye(ax, vect, point_needed=True)

        self._draw_vector_from_right_eye(ax, np.array([x, 1, z]), color="#00ff00", point_needed=True)

        arr = np.array(self.corner_points).reshape((-1, 3))
        if arr.shape[0] >= 5:
            arr[4] = arr[0]
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="#ff0000")

        # 2d part

        ax = fig.add_subplot(2, 1, 2)
        arr = np.array(self.corner_points).reshape((-1, 3))
        ax.scatter(arr[:, 0], arr[:, 2], color="#880000")
        view = self._get_plane_line_point(self.eye_right, np.array([x, 1., z]))
        ax.scatter(view[0], view[2], color="#00ff00")
        arr = np.array(self.corner_points).reshape((-1, 3))
        if arr.shape[0] >= 5:
            arr[4] = arr[0]
            ax.plot(arr[:, 0], arr[:, 2], color="#ff0000")

        # plt to img

        image = fig2img(fig)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str

    def new_corner(self, x, z):
        x, z = int(x), int(z)
        self.corner_vectors.append(np.array([x / 100, 1, z / 100]))
        self.corner_points.append(self._get_plane_line_point(self.eye_right, self.corner_vectors[-1]))
        return "kek"

    # Inner ----------------------------

    def _draw_vector_from_right_eye(self, plt_axis, vect, color="#880000", point_needed=False):
        vect = np.array(vect)
        point = self._get_plane_line_point(self.eye_right, vect)
        tmp = [self.eye_left, point, self.eye_right]
        tmp = np.array(tmp)
        plt_axis.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2], color)
        if point_needed:
            plt_axis.scatter(*point, color=color)

    def _draw_plane(self, axis):
        xx, zz = np.meshgrid(range(-150, 151, 50), range(-150, 151, 50))
        a, b, c, d = self.plane
        y = - d / b
        yy = np.full_like(xx, y)
        axis.plot_wireframe(xx, yy, zz)

    def _get_plane_line_point(self, point, vector):
        t = - np.matmul([*point, 1], self.plane) / np.matmul(vector, self.plane[0:3])
        return point + (vector * t)


if __name__ == "__main__":
    backend = BackendForDebugPredictor()
    server = zerorpc.Server(backend)
    server.bind("tcp://0.0.0.0:4242")
    server.run()
