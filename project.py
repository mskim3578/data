# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:23:30 2025

@author: EMS
"""
# pip install tensorflow

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


######### 1. 자료 수집 :   data load 
(x_train, y_train),(x_test, y_test)=\
    load_data(path='mnist.npz')

x_train.shape  # (60000, 28, 28), 훈련데이터
y_train.shape  # (60000,)
x_test.shape  # (10000, 28, 28), 테스트데이터
y_test.shape  # (10000,)

    
#image 저장 (정규화 전 자료 저장!!!!!!!)  읽어서 예측자료로 사용

tempimg = x_test[1].reshape(28,28)
tempimg
im = Image.fromarray(tempimg)
im.save("data/num.jpg", "jpeg")   

    
