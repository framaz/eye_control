from PIL import ImageTk, Image
import tkinter as tk
import numpy as np


class App:
    def __init__(self, window, window_title, button_callback):
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
        print("kek")
        btn = tk.Button(text="Hello", command=button_callback)
        btn.pack()

    # self.window.mainloop()

    def change_corner(self, corner):
        self.corner_num.text = corner
        self.corner_num.config(text=corner)

    def draw_image(self, image: Image.Image):
        if image.height > 640:
            image = image.resize((512, int(512 * image._size[1] / image._size[0])))
        img = ImageTk.PhotoImage(image)
        self.panel.image = img
        self.panel.configure(image=img)
        self.panel.pack(side="bottom", fill="both", expand="no")
        self.window.update_idletasks()
        self.window.update()

    def draw_eye(self, image, text=""):
        self.panel_text.text = text
        self.panel_text.configure(text=text)
        if not (image is None):
            img = ImageTk.PhotoImage(image)
            self.panel_eye.image = img
            self.panel_eye.configure(image=img)
        # self.panel_eye.pack(side="top", fill="both", expand="no")
        self.window.update_idletasks()
        self.window.update()
