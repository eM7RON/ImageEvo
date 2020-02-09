This was originally a uni project whereby an evolutionary algorithm was implemented to arange a collection of polygons into a close approximation of a given image. I have since started making a UI for it but it is still a work in progress. The easiest way to create a Python environment to run this code is to install anaconda and in anaconda prompt use the commands:
conda create -n ImageEvo -c defaults -c conda-forge numpy scipy scikit-learn numexpr matplotlib pillow aggdraw
conda activate ImageEvo
python -m ipykernel install --user --name ImageEvo --display-name ImageEvo

then just run the main.py file with your installed python environment.
