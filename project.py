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

#0~59999 사이의 임의의 수 3개
random_idx = np.random.randint(60000,size=3) 
for idx in random_idx :
    img = x_train[idx,:]
    label=y_train[idx] 
    plt.figure()
    plt.imshow(img)  #이미지 보기기
    plt.title\
  ('%d-th data, label is %d' % (idx,label),fontsize=15)
plt.show()     


### 2 데이터 전처리
#검증데이터 생성 : 학습 중간에 평가를 위한 데이터  
x_train,x_val,y_train,y_val = train_test_split\
    (x_train,y_train,test_size=0.3, random_state=777) 
    
    
    
# 2-1데이터 정규화
'''
  MinMax normalization : X = (x-min)/(max-min)
  Robust mormalization : X=(x-중간값)/(3분위값-1분위값)
  Standardization      : X=x-평균값/표준편차
'''
x_train[0]
#MinMax normalization 정규화
#현재데이터 : min:0, max=255
x_train = (x_train.reshape(42000,28*28))/255 
x_val = (x_val.reshape(18000,28*28))/255
x_test = (x_test.reshape(10000,28*28))/255
x_train[0]
x_train.shape #(42000, 784)
x_val.shape   #(18000, 784)
x_test.shape  #(10000, 784)
y_train[:10]

# 2-2  레이블 전처리:one-hot 인코딩하기 multi classfication

y_train=to_categorical(y_train)
y_train[:10]
y_val=to_categorical(y_val)
y_test=to_categorical(y_test)



# 3 학습 모델 구성하기

model = Sequential()  #모델 생성
model.add(Dense(64,activation="relu",input_shape=(784,)))
model.add(Dense(32,activation="relu"))
model.add(Dense(10,activation="softmax"))

