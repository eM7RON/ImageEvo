import sys
import os
import time
import re

from PyQt5.QtCore import QThread
import numpy as np


def qimage_to_array(img):
    ''' 
    Converts a QImage to a 3d numpy array
    Args:
        img - PyQt.QtGui.QImage
    Returns:
        arr - numpy.ndarray, shape(h, w, 4)
    '''
    img = img.convertToFormat(4)
    w = img.width()
    h = img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.array(ptr).reshape(h, w, 4)[:, :, [2, 1, 0, 3]] # BGRA -> RGBA
    return arr

def directory_explorer(extension, directory, return_abspath=False):
    '''
    A generator to find filenames with a given extension within a given 
    directory
    '''
    extension = extension.lower()
    for fname in os.listdir(os.path.normpath(directory)):
        if fname.lower().endswith(extension):
            yield os.path.abspath(fname) if return_abspath else fname

def atoi(string: str):
    '''
    Asscii To Int:
        Takes a str and returns an int if it is made of digits, 
        str otherwise
    '''
    return int(string) if string.isdigit() else string

def natural_order(string: str) -> list:
    '''
    A sorting key:
    Uses:
        alist.sort(key=natural_order) or sorted(alist, key=natural_order)
        sorts in human order.
    How it works:
        We are trying to sort a list (x) of strings in natural sorting order.
        A string is split into substrings between where regions of digits and
        non-digits occur. atoi(substring) converts digits to type=int whereas
        non-digit substrings remain as str. This creates a list for each string
        in x. The nested lists are then sorted by the default lexi-sort key 
        used by Python.
    Returns:
        list - containing, a priori, unknown number or order of str and/or ints
    '''
    return [atoi(substr) for substr in re.split(r'(\d+)', string)]

def fit_to_screen(screen, img):
    '''
    Scales the dimensions of an image to fit a screen resolution
    whilst preserving the aspect ratio and causing no stretching
    
    Args:
        screen   - 2-tuple, (width, height) of the target screen 
                   resolution
        img      - 2-tuple, (width, height) of the image to be rendered
                   on the screen
    '''
    # Aspect ratios
    aspr_s = screen[0] / screen[1]
    aspr_i = img[0] / img[1]
    if aspr_s > aspr_i:
        scale_factor = screen[1] / img[1]
    else:
        scale_factor = screen[0] / img[0]
    return round(img[0] * scale_factor), round(img[1] * scale_factor)

def get_svg_dimensions(fname: str):
    '''
    All SVGs made with this software have the viewbox anchored in the top left corner
    (0, 0). This means that the viewBox coordinates end up containing the original image
    dimensions. This funtion harvests those dimensions.
    Returns:
        tuple[int, int] - (width, height)
    '''
    with open(fname, 'r') as open_file:
            svg_string = open_file.read()
    return tuple(int(x) for x in re.search(r'viewBox="\d \d (\d+ \d+)"', svg_string).group(1).split())

def safely_create_directory(where: str, name: str='temp', return_required: bool=True):
    '''
    Create a new directory without overwriting
    any prexisting directories
    '''
    new_dir = os.path.join(where, name)
    exists = os.path.isdir(new_dir) # True if dir exists
    if exists:
        i = 0
        while exists:
            i += 1
            exists = os.path.isdir(new_dir + f' ({i})')
        new_dir += f' ({i})'
    os.mkdir(new_dir)
    if return_required:
        return new_dir

def safely_create_fname(where: str, fname: str='temp') -> str:
    '''
    Create a new fname without overwriting
    any prexisting directories
    '''
    new_fname = os.path.join(where, fname)
    exists = os.path.isfile(new_fname) # True if dir exists
    if exists:
        i = 0
        while exists:
            i += 1
            exists = os.path.isfile(new_fname + f' ({i})')
        new_fname += f' ({i})'
    return new_fname

def bit_max(v: int) -> int:
    '''
    Calculates number of bits required to represent the all of the values between 
    0 and v
    '''
    u = 1
    while int("1" * u, 2) < v:
        u += 1
    return u

def replace_extension(fname: str, extension: str) -> str:
    fname = fname.split('.')[0]
    return f'{fname}.{extension}'

class EventLoopThread(QThread):
    '''
    A custom thread class which runs its own event loop and therefore
    can accept signals from the master thread
    '''
    def run(self):
        self.exec_()