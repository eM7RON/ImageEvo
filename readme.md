# ImageEvo

Some software that allows a user to arrange a collection of randomly generated shapes into a close approximation of a given image. Currently, only one algorithm is implemented (Geometric Particle Swarm Optimization) but others will be added. The project started as some university coursework to come up with a novel application for the GPSO algorithm. Eventually, the algorithm was implemented in a GUI for fun!

## Getting Started

### Prerequisites

python 3  
numpy  
scipy  
scikit-learn  
pillow  
aggdraw  
numexpr  
pyqt 

### Installing

The easiest way to create a Python environment to run this code is to install anaconda and in anaconda prompt use the commands:

`conda create -n ImageEvo -c defaults -c conda-forge numpy scipy scikit-learn numexpr matplotlib pillow aggdraw`

`conda activate ImageEvo`

OpenCV is also a requirment. On windows it is best to obtain it from here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv

then navigate to the download and install it with pip:

`pip install <path/to/whl>`

Otherwise on Mac/Linux it's possible to use:

`pip install opencv-python`

Then just download the repository, extract, navigate to the directory with main.py and run main.py with the previously installed python environment:

`python main.py`

Although `ImageEvo` does offer some video rendering abilities, users are responsible for supplying video codecs. A popular H264 codec is available here: https://github.com/cisco/openh264/releases

Another option is to build `opencv`, linked with `ffmpeg`, from source using `cmake`.

### Instructions

The software launches to a `Main Menu` which allows the user to select between evolving a new image (by choosing an algorithm e.g. `GPSO`), loading up a previously initialized population (`Load`), editing and image (`Image Editor`) or creating a video (`Video Maker`).

<p align="center">
<img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/tour/main_menu.svg" alt="Main Menu" width="700"/>
</p>

If an algorithm is selected e.g. `GPSO`, a setup screen opens where the user can choose from various parameters and run the algorithm. The background colors of the text inputs indicate the validity of the input i.e. red=invalid, yellow=valid (with warning), green=valid. The user can choose between different primitive shape types: circles, ellipses, squares, rectangles and polygons. Polygons can have a user defined number of vertices.

<p align="center">
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/tour/gpso_setup.svg" alt="GPSO Setup" width="1100"/>
</p>

When the an algorithm is run, the progress will be displayed in a separate window as below. The current 'best' solution will be displayed in a window on the left (`progress image`). The image format is SVG and the image will be fit to the window whilst preserving the original aspect ratio. However, the `progress image` may appear larger than the orignal image. A `matplotlib` figure is display in a widget to the lower left. This shows the progress over time. Depending on the selected algorithm, this figure can display different series e.g. standard deviation or mean, which are easy to toggle on and off. Some stats will be displayed above this figure such as fitness and iterations etc... At the bottom right there are controls to `pause` the algorithm and `save` progress. These controls may seem 'sluggish' but bare in mind that the controls are toggling a flag and the selected action is performed at a specific location in the main loop.

<p align="center">
     <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/tour/gpso.svg" alt="GPSO" width="1150"/>
</p>

The size of the input image will affect the speed of the algorithm. Images of around 256×256 pixels are perfect. The `Image Editor`:

<p align="center">
<img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/tour/image_editor.svg" alt="Image Editor" width="900"/>
</p>

can be accessed from the `Main Menu` and has the ability to resized images. It is a basic API for some of Python pillow's functionality allowing some filtering via PIL.ImageFilter.

A running algorithm will output images into a chosen folder at chosen iteration intervals. The format of the images is SVG. This means they take up little space and are lossless. The `Video Maker`, when pointed at a folder with the aforementioned SVG images, can convert them into a video. `Video Maker` allows the user to select any codecs and containers supported by `cv2`. However, it is up to the user to make sure the codecs are available. A freely available codec can be downloaded from https://github.com/cisco/openh264/releases. Just download the correct one for your OS, extract it and place it in the same directory as your python.exe. Version 1.8.0 was required for Python 3.8.

Here are some combinations which worked for me:

| Codec | Container |
|-------|-----------|
| avc3  | mp4       |
| avc1  | mov       |
| x264  | mkv       |

<p align="center">
     <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/tour/video_maker_setup.svg" alt="Video Maker Setup" width="900"/>
</p>

Most video formats will handle approximately 60 fps maximum, and if you have generated millions of images, this may mean your video will be too long. Furthermore, because most of the improvement of the algorithm occurs near the start, towards the end there may be thousands of images which are the same, and little changes, which can lead to a boring video. To alleviate this, `Video Maker` includes some sampling methods which can allow the user to create a video from a subset of the images in the supplied folder. For example, the user may want to use `Exponential decay forward` which will sample early images much more frequent than later images, where less canged occurs. This creates a video where improvement is more constant and more interesting to watch. It is also possible to create a 'reversed' video where fitness decreases over time and the 'best' image achieved may decay into a collection of random images.

The video rendering progress is displayed as below:

<p align="center">
     <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/tour/video_maker.svg" alt="Video Maker" width="380"/>
</p>

### Examples

The examples below were generated using `GPSO` with a single starting image and an increasing number of images. This was found to generate the best images with GPSO.

<p align="center">
     <b> Master Chief </b>
</p>

<p align="center">
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/chief/110.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/chief/810.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/chief/3560.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/chief/9460.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/chief/26860.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/chief/78760.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/chief/268010.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/chief/723880.svg" alt="Video Maker" width="200"/>
</p>

<p align="center">
     <b> Jimi Hendrix </b>
</p>

<p align="center">
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/hendrix/1930.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/hendrix/7980.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/hendrix/12920.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/hendrix/24440.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/hendrix/45350.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/hendrix/100270.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/hendrix/186510.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/hendrix/280630.svg" alt="Video Maker" width="200"/>
</p>

<p align="center">
     <b> Skull </b>
</p>

<p align="center">
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/skull/150.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/skull/5510.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/skull/13150.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/skull/28700.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/skull/67550.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/skull/143350.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/skull/255600.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/skull/286070.svg" alt="Video Maker" width="200"/>
</p>

<p align="center">
     <b> Walter White </b>
</p>

<p align="center">
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/walter/130.svg" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/walter/1550.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/walter/11650.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/walter/25110.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/walter/57380.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/walter/121040.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/walter/215190.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/demo/walter/343420.svg" alt="Video Maker" width="200"/>
</p>

## Upcoming Improvements

Eventually I will add other algorithms such as a conventional genetic algorithm. Python is, perhaps, not the best language for this type of thing as it seems quite slow. I have profiled the GPSO algorithm and the bottle neck seems to be the image rendering. This could be eleviated by writing a custom wrapper for the C++ agg library. This would probably take a 2D array as input and parallelize the image rendering by dividing the individuals between threads. Another addition to the software would be to be able to use multiple different primitive shape types.

## Authors

* **eM7RON (Simon Tucker)** - *Initial work* - [github](https://github.com/eM7RON), [linkedin](https://www.linkedin.com/in/simon-tucker-21838372/)

## License

This project is licensed under the MIT License

## Acknowledgments

* **Moraglio, A., Chio, C.D., Toggelius, J., and Poli, R.,** - *Geometric Particle Swarm Optimization. Journal of Artiﬁcial Evolution and Applications. 2008.* - [http://dx.doi.org/doi:10.1155/2008/143624](http://dx.doi.org/doi:10.1155/2008/143624)
* **Freepik** - [*DNA Icon*](https://www.flaticon.com/free-icon/dna_620330?term=dna%20freepik&page=1&position=6), [*Paint Image*](https://www.flaticon.com/free-icon/paint_1157969?term=paint%20palette&page=1&position=6) 
* **Vectors Markets** - [*Pause Icon*](https://www.flaticon.com/free-icon/pause_165602?term=pause&page=3&position=94)
* **DinosoftLabs** - [*TV Icon*](https://www.https://www.flaticon.com/free-icon/tv_716429)
* **Becris** - [*Alg Image*](https://www.flaticon.com/free-icon/neural_2103633?term=algorithm&page=1&position=6)
