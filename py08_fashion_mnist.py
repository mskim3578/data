# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 11:33:38 2025

@author: letuin
"""
# pip install tensorflow
# import tensorflow as tf

##################################################
# Fashion-MNIST 
from tensorflow.keras.datasets.fashion_mnist import load_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 수집 :  set_loaddata()
# 2. 전처리 ???? : pre_dataprocess()
# 3. model 만들기 : set_model()
# 4. 평가 : pro_evaluation()
# 5. 예측 : pro_predict()

def set_loaddata():
    global originx_train, originy_train,originx_test,originy_test
    (originx_train, originy_train), (originx_test, originy_test) = load_data()
    
def pre_dataprocess(): 
    global modelx_train,modelx_val,modely_train,modely_val
    modelx_train,modelx_val,modely_train,modely_val = train_test_split\
        (originx_train,originy_train,test_size=0.3, random_state=777)  
    
    #  x(변수)  min-max normalization : (x-min)/(max-min) 이미지 x/255
    # modelx_train.shape #(42000, 28, 28)
    modelx_train = (modelx_train.reshape(42000,28*28))/255 
    modelx_val = (modelx_val.reshape(18000,28*28))/255
    
    # y(label) one-hot encoding
    modely_train=to_categorical(modely_train)
    modely_val=to_categorical(modely_val)
    
    # y 종류  10가지 ---> multi classfication
    # 7: --->  00000 00100
    # 3: --->  00010 00000
    # 5: --->  00000 10000

def set_model():
    model=Sequential()
    model.add(Dense(64,activation="relu",input_shape=(784,)))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(10,activation="softmax"))

    model.compile(optimizer="adam", loss='categorical_crossentropy',
                   metrics=['acc'])
    
    # 학습
    history=model.fit(modelx_train,modely_train,epochs=30,batch_size=127,
                      validation_data=(modelx_val,modely_val)) 
    
    return model, history

def pro_evaluation(model,history):    
    # mx_test=originx_test.reshape(10000, 28*28)/255 
    # my_test=to_categorical(originy_test)  
    model.evaluate(modelx_val,modely_val) 
    
    his_dict = history.history
    loss = his_dict['loss']  #훈련데이터 학습시 손실함수값
    acc = his_dict['acc'] #훈련데이터 정확도값
    val_loss = his_dict['val_loss'] #검증데이터 학습시 손실함수값
    val_acc = his_dict['val_acc'] #검증데이터 정확도값
    
    epochs = range(1, len(loss) + 1) #1 ~ 30까지의 숫자
   
    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(1, 2, 1) #1행2열의 1번째 그래프영역
    ax1.plot(epochs , loss, color="blue", label="train_loss")
    ax1.plot(epochs , val_loss, color="orange", label="val_loss")
    ax1.set_title('train and val loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend()
    
    ax2 = fig.add_subplot(1, 2, 2) #1행2열의 1번째 그래프영역
    ax2.plot(epochs , acc, color="blue", label="train_acc")
    ax2.plot(epochs , val_acc, color="orange", label="val_acc")
    ax2.set_title('train and val acc')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('acc')
    ax2.legend()
    plt.show()


def pro_predict(model, rangelist, all=True):
    global results, results_y
    # x normal
    mx_test=originx_test.reshape(10000, 28*28)/255 
    results = model.predict(mx_test)
    results_y=np.argmax(results, axis=1)
    count=0
    for idx in rangelist:
        print(idx)
        pred_y=results_y[idx]
        label_y=originy_test[idx]  
        if pred_y != label_y or all:
            plt.subplot(4, 4, count+1)  #4행4열
            plt.imshow(originx_test[idx]) #2차원배열. 그래프
            plt.title('idx:%d \n Pred:%s,\n lab:%s' % 
                      (idx, class_names[pred_y],class_names[label_y])
                      ,fontsize=8)
            count +=1
            if count > 15 : break
    plt.tight_layout()
    plt.show() 

from PIL import Image
def image_save(num):        
    tempimg = originx_test[num]
    im = Image.fromarray(tempimg)
    im.save(f"img/fashion{num}.jpg", "jpeg") 

   
image_save(127)    
    
results_y[:10]    
originy_test[:10]    
    
set_loaddata()    
pre_dataprocess()  
model,history=set_model() 


his_dict = history.history
pro_evaluation(model,history)


class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# def pro_predict(model, rangelist, all=True):
pro_predict(model, range(len(originx_test)), False) # 틀린예측
pro_predict(model, range(60,76), True)  # 전체   

model.save('flask_number/fashion.keras')



    