import typing
from PIL import ImageTk, Image
import tkinter as tk
import numpy as np


class App:
    """Class for tkinter activity

    """
    def __init__(self, window: tk.Tk, window_title: str, button_action: typing.Callable):
        """Constructor

        :param window: tk.Tk window
        :param window_title:
        :param button_action: button callback function
        """
        self.window = window
        self.window.title(window_title)
        self.window.update_idletasks()
        self.window.update()

        self.img = Image.fromarray(np.ones((500, 500)))
        self.panel = tk.Label(self.window)
        self.panel.pack(side="bottom", fill="both", expand="no")
        self.panel_eye = tk.Label(self.window)
        self.panel_eye.pack(side="top", fill="both", expand="no")
        self.panel_text = tk.Label(self.window)
        self.panel_text.pack(side="top", fill="both", expand="no")
        self.corner_num = tk.Label(self.window)
        self.corner_num.pack(side="top", fill="both", expand="no")

        btn = tk.Button(text="Hello", command=button_action)
        btn.pack()

    # self.window.mainloop()

    def change_corner(self, corner: str) -> None:
        """Set text for current corner

        :param corner: corner text
        :return:
        """
        self.corner_num.text = corner
        self.corner_num.config(text=corner)

    def draw_image(self, image: typing.Union[Image.Image, np.ndarray], max_size="small") -> None:
        """Draws downscaled image

        :param image:
        :param max_size: can be "small", 'medium', "large"
        :return:
        """
        if not (isinstance(image, Image.Image)):
            image = image.astype(dtype=np.uint8)
            image = Image.fromarray(image)

        max_size = {"small": 512, "medium": 1024, "large": 1536, "no": 10000000}[max_size]
        if image.height > max_size:
            image = image.resize((max_size, int(max_size * image._size[1] / image._size[0])))

        img = ImageTk.PhotoImage(image)
        self.panel.image = img
        self.panel.configure(image=img)
        self.panel.pack(side="bottom", fill="both", expand="no")

        self.window.update_idletasks()
        self.window.update()

    def draw_eye(self, image: typing.Union[Image.Image, None], text: typing.Any = "") -> None:
        """Draws eye and text

        :param image:
        :param text:
        :return:
        """
        self.panel_text.text = str(text)
        self.panel_text.configure(text=text)

        if not (image is None):
            img = ImageTk.PhotoImage(image)
            self.panel_eye.image = img
            self.panel_eye.configure(image=img)
        # self.panel_eye.pack(side="top", fill="both", expand="no")

        self.window.update_idletasks()
        self.window.update()


cycling_flag = True


def button_callback():
    global cycling_flag
    cycling_flag = False
    print("azazaza")
