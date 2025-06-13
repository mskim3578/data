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

model.compile(optimizer="adam", loss='categorical_crossentropy',
              metrics=['acc'])

'''
optimizer="adam" : 경사하강법 알고리즘 이름.
                   Adam 클래스로도 가능 => import 해야함
loss='categorical_crossentropy'  : 손실함수 종류. 
                   label(정답) ont-hot 인코딩 되어야함                  
     mse : 평균제곱오차.
     categorical_crossentropy : 다중분류에서 사용되는 손실함수
            => 활성화함수 : softmax 와 보통 같이 사용됨
     binary_crossentropy : 이진분류에서 사용되는 손실함수
            => 활성화함수 : sigmoid 와 보통 같이 사용됨
metrics=['acc'] : 평가지표.            
'''

# 4-1  학습하기
history=model.fit(x_train,y_train,epochs=30,batch_size=127,
                  validation_data=(x_val,y_val))



# 5. 검증
history.history["loss"] #훈련데이터 손실함수값
len(history.history["loss"])
history.history["acc"] #훈련데이터 정확도
history.history["val_loss"] #검증데이터 손실함수값
history.history["val_acc"] #검증데이터 정확도값


his_dict = history.history
loss = his_dict['loss']  #훈련데이터 학습시 손실함수값
val_loss = his_dict['val_loss'] #검증데이터 학습시 손실함수값
epochs = range(1, len(loss) + 1) #1 ~ 30까지의 숫자
fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(1, 2, 1) #1행2열의 1번째 그래프영역
ax1.plot(epochs, loss, color = 'blue', label = 'train_loss')
ax1.plot(epochs, val_loss, color = 'orange', label = 'val_loss')
ax1.set_title('train and val loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend() 
#정확도 그래프
acc = his_dict['acc'] #훈련데이터 정확도값
val_acc = his_dict['val_acc'] #검증데이터 정확도값
ax2 = fig.add_subplot(1, 2, 2) #1행2열의 2번째 그래프 영역
ax2.plot(epochs, acc, color = 'blue', label = 'train_acc')
ax2.plot(epochs, val_acc, color = 'orange', label = 'val_acc')
ax2.set_title('train and val acc')
ax2.set_xlabel('epochs')
ax2.set_ylabel('acc')
ax2.legend() 
plt.show()

'''
   과적합현상 발생 : 훈련을 너무 많이함.
            훈련을 해도 검증 데이터의 평가지수가 개선 안됨.
'''
loss[29]     #0.00736351078376174
val_loss[29] #0.14246465265750885
acc[29]      #0.9978333115577698
val_acc[29]  #0.9711111187934875
#모델 평가
#[0.13174931704998016, 0.9735000133514404]
#[손실함수값, 정확도]
model.evaluate(x_test,y_test) 

# 6 예측
results = model.predict(x_test)



tempimg = x_test[300].reshape(28,28)
tempimg
im = Image.fromarray(tempimg)
im.save("data/num300.jpg", "jpeg")   

#read image 예측
readimage = Image.open('data/num300.jpg')  #2
numpyimage = np.asarray(readimage) 

#numpy shape setting
numpyimage.shape
normalimage=numpyimage/255  # 정규화화

numpyimage=numpyimage.reshape(1,28*28)   # predict 파라메타를 위한 shape 적용

#predict를 위한자료로 reshape  (1, 784)

onetest = model.predict(numpyimage) 


np.argmax(onetest,axis=1)[0] #2
plt.imshow(numpyimage.reshape(28, 28)) #2차원배열. 그래프
plt.show()   #이미지 view








