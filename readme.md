# Image Evolution

Some software that allows a user to evolve an image by arranging a collection of randomly generated polygons into a close approximation of a given image. Currently, only one algorithm is implemented (Geometric Particle Swarm Optimization) but others will be added. The project started as some university coursework to come up with a novel application for the GPSO algorithm. Eventually, the algorithm was implemented in a GUI for fun!

## Getting Started

Just download the repository and run the main.py file with the python environment installed as outlined below.

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

conda create -n ImageEvo -c defaults -c conda-forge numpy scipy scikit-learn numexpr matplotlib pillow aggdraw

conda activate ImageEvo

python -m ipykernel install --user --name ImageEvo --display-name ImageEvo

then just download the repository and run the main.py file with the previously installed python environment.

## Authors

* **eM7RON (Simon Tucker)** - *Initial work* - [github](https://github.com/eM7RON), [linkedin](https://www.linkedin.com/in/simon-tucker-21838372/)

## License

This project is licensed under the MIT License
