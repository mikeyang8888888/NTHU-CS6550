Run hw1_1.py

This python file will generate a file named process_results which contained original and transformed images.

#---------------------

Library:
import os
import cv2
import math
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import skimage

#---------------------

adjust kernel size:

def run(img, process, gaussian_size = xx, s_tensor_windowSize = yy):

gaussian_size --> Gaussian smooth kernel size
s_tensor_windowSize --> window size of structure tensor