
import tensorflow as tf

from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from PIL import Image
from keras.models import load_model
from sklearn.metrics import \
    classification_report,confusion_matrix
    
