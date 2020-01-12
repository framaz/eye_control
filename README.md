# Eye control

Eye control is a project that allows you to use your gaze to move mouse cursor.

## Requierements

* python>=3.6
* Tensorflow 2
* dlib
* openCV
* pillow (PIL)
* numpy
* matplotlib
* tkinter
* Jupyter notebook (if you want to train your NN yourself)

## Project structure

The project is **divided into 2 parts**

* NN supervised learning. It is located in /, the main files are some .ipynb. Currently the project is using [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/), you can download it or use different dataset.
* Watcher - gaze estimation to control mouse cursor. It is located in /watcher/.

## Running the watcher

At the start of the program you have to calibrate it.

1. Run calibration_pic in fullscreen. If your screen isnt 1920x1080 then draw calibration pic yourself or wait for future versions.

2. Run the program.

Next step you have to do with **fixed head position**, dont move your head etc.

3. Concentrate your view on the point for 5-10 seconds without closing your eyes then press the button. Repeat this with all points.

4. Use the program.

**If you move the camera or the screen then you have to restart the program**

## Special thanks

*  yinguobing for pnp solve module from [Head pose estimation project](https://github.com/yinguobing/head-pose-estimation) 
