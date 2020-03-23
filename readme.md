# ImageEvo

Some software that allows a user to evolve an image by arranging a collection of randomly generated polygons into a close approximation of a given image. Currently, only one algorithm is implemented (Geometric Particle Swarm Optimization) but others will be added. The project started as some university coursework to come up with a novel application for the GPSO algorithm. Eventually, the algorithm was implemented in a GUI for fun!

## Getting Started

### Prerequisites

python 3  
numpy  
scipy  
scikit-learn  
pillow  
aggdraw  
numexpr  

### Installing

The easiest way to create a Python environment to run this code is to install anaconda and in anaconda prompt use the commands:

conda create -n ImageEvo -c defaults -c conda-forge ipykernel numpy scipy scikit-learn numexpr matplotlib pillow aggdraw

conda activate ImageEvo

python -m ipykernel install --user --name ImageEvo --display-name ImageEvo

then just download the repository and run the main.py file with the previously installed python environment.

### Instructions

The software launches to a `Main Menu` which allows the user to select between evolving a new image (by choosing an algorithm e.g. `GPSO`), loading up a previously initialized population (`Load`), editing and image (`Image Editor`) or creating a video (`Video Maker`).

<p align="center">
<img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/main_menu.PNG" alt="Main Menu" width="300"/>
</p>

If an algorithm is selected e.g. `GPSO`, a setup screen opens where the user can choose from various parameters and run the algorithm. The background colors of the text inputs indicate the validity of the input i.e. red=invalid, yellow=valid (with warning), green=valid.

<p align="center">
<img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/gpso_setup.PNG" alt="GPSO Setup" width="300"/>
</p>

The size of the input image will affect the speed of the algorithm. Images of around 256×256 pixels are perfect. The `Image Editor`:

<p align="center">
<img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/image_editor.PNG" alt="Image Editor" width="300"/>
</p>

can be accessed from the `Main Menu` and has the ability to resized images. It is a basic API for some of Python pillow's functionality allowing some filtering via PIL.ImageFilter.

A running algorithm will output images into a chosen folder at chosen iteration intervals. The format of the images is SVG. This means they take up little space and are lossless. The `Video Maker`, when pointed at a folder with the aforementioned SVG images, can convert them into a video. `Video Maker` allows the user to select any codcs and containers supported by `cv2`. However, it is up to the user to make sure the codecs are available.

<p align="center">
<img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/video_maker.PNG" alt="Video Maker" width="300"/>
</p>

### Examples

The examples below were generated using `GPSO` with a single starting image and an increasing number of images. This was found to generate the best images with GPSO.

<p align="center">
     <b> Master Chief <b/>
<p/>

<p align="center">
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/chief/110.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/chief/810.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/chief/3560.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/chief/9460.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/chief/26860.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/chief/78760.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/chief/268010.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/chief/461810.svg" alt="Video Maker" width="200"/>
</p>

<p align="center">
     <b> Jimi Hendrix <b/>
<p/>

<p align="center">
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/hendrix/1930.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/hendrix/7980.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/hendrix/12920.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/hendrix/24440.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/hendrix/45350.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/hendrix/100270.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/hendrix/186510.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/hendrix/280630.svg" alt="Video Maker" width="200"/>
</p>

<p align="center">
     <b> Skull <b/>
<p/>

<p align="center">
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/skull/150.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/skull/5510.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/skull/13150.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/skull/28700.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/skull/67550.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/skull/143350.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/skull/255600.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/skull/286070.svg" alt="Video Maker" width="200"/>
</p>

<p align="center">
     <b> Walter White <b/>
<p/>

<p align="center">
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/walter/130.svg" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/walter/1550.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/walter/11650.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/walter/25110.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/walter/57380.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/walter/121040.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/walter/215190.svg" alt="Video Maker" width="200"/>
    <img src="https://github.com/eM7RON/Image-Evolution/blob/master/img/walter/343420.svg" alt="Video Maker" width="200"/>
</p>

## To Do

Eventually I will add other algorithms such as a conventional genetic algorithm.
Python is, perhaps, not the best language for this type of thing as it seems quite slow. I have profiled the GPSO algorithm and the bottle neck seems to be the image rendering. This could be eleviated by writing a custom wrapper for the C++ agg library. This would probably take a 2D array as input and parallelize the image rendering by dividing the individuals between threads.

## Authors

* **eM7RON (Simon Tucker)** - *Initial work* - [github](https://github.com/eM7RON), [linkedin](https://www.linkedin.com/in/simon-tucker-21838372/)

## License

This project is licensed under the MIT License

## Acknowledgments

* **Moraglio, A., Chio, C.D., Toggelius, J., and Poli, R.,** - *Geometric Particle Swarm Optimization. Journal of Artiﬁcial Evolution and Applications. 2008.* - [DOI](http://dx.doi.org/doi:10.1155/2008/143624)
* **Freepik** - *DNA Icon* - [www.flaticon.com](https://www.flaticon.com/free-icon/dna_620330?term=dna%20freepik&page=1&position=6)
* **Vectors Markets** - *Pause Icon* - [www.flaticon.com](https://www.flaticon.com/free-icon/pause_165602?term=pause&page=3&position=94)


