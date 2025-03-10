import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
# fashion_mnist = keras.datasets.fashion_mnist
from matplotlib.markers import MarkerStyle 
from keras import backend as K
from keras.optimizers import Adam
from keras.datasets import mnist 
from keras.layers import Lambda, Input, Dense 
from keras.losses import binary_crossentropy 
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from glob import glob
from PIL import Image
from time import time
from sklearn.model_selection import train_test_split
import os
import imageio
from IPython.display import Image as Img
