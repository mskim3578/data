
import pandas as pd
pd.__version__
'''
cmd 창에서 실행 한다  (vscode)


pip install tensorflow


'''

#MNIST 데이터를 이용하여 숫자를 학습하여 숫자 인식하기.
#MNIST 데이터셋 다운받기
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
#from tensorflow.keras.datasets.fashion_mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from PIL import Image
from keras.models import load_model
from sklearn.metrics import confusion_matrix

# 1. load_data()
# 2. model setting을 이용한 전처리 
# 3. setModel (model, compile, fit)
# 4. 평가
# 5. 예측


'''
  #number mnist
  #plt.title('Pred:%d,lab:%d' % (pred_y,label_y),fontsize=15)
  
  #fashine mnist
  #plt.title('Pred:%s,\n lab:%s' % (class_names[pred_y],class_names[label_y]),fontsize=15)


'''

######### 1. 자료 수집 :   data load  
def set_loaddata():
    global originx_train, originy_train, originx_test, originy_test
    # numbermnist
    # (originx_train, originy_train),(originx_test, originy_test)\
    # =load_data(path='mnist.npz')
    
    # fashine mnist
    (originx_train, originy_train), (originx_test, originy_test) = load_data()
    
######### 2  데이터 전처리
def pre_dataprocess():
    global modelx_train,modelx_val,modely_train,modely_val
    # 2-1 검증데이터 생성 : 학습 중간에 평가를 위한 데이터  
    modelx_train,modelx_val,modely_train,modely_val = train_test_split\
        (originx_train,originy_train,test_size=0.3, random_state=777)  
    
    # 2-2 독립변수 (x) 전처리  : min-max normalization
    '''
      MinMax normalization : X = (x-min)/(max-min)
      Robust mormalization : X=(x-중간값)/(3분위값-1분위값)
      Standardization      : X=x-평균값/표준편차
    '''
    modelx_train = (modelx_train.reshape(42000,28*28))/255
    modelx_val = (modelx_val.reshape(18000,28*28))/255
     # 2-3  레이블(y) 전처리:one-hot 인코딩하기 multi classfication
    modely_train=to_categorical(modely_train)
    modely_val=to_categorical(modely_val)
      
 

def set_model():
    # 3 학습 모델 구성하기

    # 지도 학습 model matching
    '''
     1. binary classfication
       final activation  : softmax
       optimizer : adm : 경사 하강법
       loss : binary_crossentropy
     
     2. multi classfication
       final activation  : softmax
       optimizer : adm : 경사 하강법
       loss : categorical_crossentropy
       
     3. linear classfication
       final activation  : linear (기본값)
       optimizer : mse : 경사 하강법
       loss : mean_squared_error (MSE) 평균 제곱오차 
       
    '''




    model = Sequential()  #모델 생성
    model.add(Dense(64,activation="relu",input_shape=(784,)))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(10,activation="softmax"))

    '''
      
      softmax : 값들간의 영향을 줌. 다중분류에서 많이 사용됨
                  결과값의 합은 1  
      sigmoid : 값들간의 영향 없음. 0~1사이의 값.   
      relu :  0보다작으면 0 크면 그값을 유지한다 
               
    '''
    '''
      1층 : 
        64 : 출력노드 갯수   
        input_shape=(784,) : 입력노드의 갯수
        activation="relu" : 활성화 함수. 0이상의 값
      2층 :
        32 : 출력노드 갯수  
        activation="relu" : 활성화 함수. 0이상의 값
        입력노드갯수 : 1층의 출력노드갯수.64개
      3층 :
        10 : 출력노드 갯수. 0~9까지의 수. 다중분류 모델  
        activation="softmax" : 활성화 함수. 
                        다중분류 방식에서 사용되는 활성화 함수
        입력노드갯수 : 2층의 출력노드갯수.32개
    '''
    model.summary()
    '''
    Param # : 가중치 편향의 갯수
    1층 : (784 + 1) * 64 = 50240
    2층 : (64 + 1) * 32 = 2080
    3층 : (32 + 1) * 10 = 330
    Total params: 52,650
    '''


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
    # 4   학습하기
    history=model.fit(modelx_train,modely_train,epochs=30,batch_size=127,
                      validation_data=(modelx_val,modely_val)) 

    
    '''
    epochs=30 : 30번 학습하기.
    batch_size=127 : 데이터를 127개로 분리.기본값:32
                 42000/127=330.7086614173228
    validation_data=(x_val,y_val) : 검증데이터 설정. 
    history : 학습 과정을 저장한 데이터            
    '''
    return model ,history

def evaluation(model, history):
     
    # x: normalization
    modelx_test=originx_test.reshape(10000, 28*28)/255 
    # y: one-hot encoding
    modely_test=to_categorical(originy_test)  
  
    model.evaluate(modelx_test,modely_test) 
    # 검증 자료 확인 
    history.history["loss"] #훈련데이터 손실함수값
    len(history.history["loss"])
    history.history["acc"] #훈련데이터 정확도
    history.history["val_loss"] #검증데이터 손실함수값
    history.history["val_acc"] #검증데이터 정확도값
    type(history.history) #dict

    # 4-2  결과 시각화 하기


    his_dict = history.history
    loss = his_dict['loss']  #훈련데이터 학습시 손실함수값
    acc = his_dict['acc'] #훈련데이터 정확도값
    val_loss = his_dict['val_loss'] #검증데이터 학습시 손실함수값
    val_acc = his_dict['val_acc'] #검증데이터 정확도값
    loss[29]     #0.00736351078376174
    val_loss[29] #0.14246465265750885
    acc[29]      #0.9978333115577698
    val_acc[29]  #0.9711111187934875

    '''
       과적합현상 발생 : 훈련을 너무 많이함.
                훈련을 해도 검증 데이터의 평가지수가 개선 안됨.
    '''



    # loss 시각화
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
    ax2 = fig.add_subplot(1, 2, 2) #1행2열의 2번째 그래프 영역
    ax2.plot(epochs, acc, color = 'blue', label = 'train_acc')
    ax2.plot(epochs, val_acc, color = 'orange', label = 'val_acc')
    ax2.set_title('train and val acc')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('acc')
    ax2.legend() 
    plt.show()


    
def predict_pro(model, rangelist, all=True):
    modelx_test=originx_test[rangelist].reshape(-1, 28*28)/255 
    results=model.predict(modelx_test)
    results_y=np.argmax(results, axis=1)
    print(results_y)
    count=0
    
    # fashinemnist
    class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
    
    for idx in range(len(rangelist)):
        pred_y = results_y[idx]
        label_y = originy_test[idx]
        if label_y != pred_y or all : 
            
            plt.subplot(4, 4, count+1)  #4행4열
            plt.axis('off') 
            plt.imshow(originx_test[idx].reshape(28, 28)) #2차원배열. 그래프
            plt.title('Pred:%d,lab:%d' % (pred_y,label_y),fontsize=15)
            
            #fashine mnist
            #plt.title('Pred:%s,\n lab:%s' % (class_names[pred_y],class_names[label_y]),fontsize=15)
      
            count +=1
            if count > 15 : break
    plt.tight_layout()
    plt.show()  
   
    # 혼동행렬 조회하기
    # 실제 값과 예측값의 비교를 위한 행렬
    cm=confusion_matrix(originy_test[rangelist], results_y) 
                       
                        
    #heatmap으로 출력하기
    import seaborn as sns
    plt.figure(figsize=(7,7))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.xlabel('predicted label',fontsize=15)
    plt.ylabel('true label',fontsize=15)
    plt.show()
        

def image_save(num):        
    tempimg = originx_test[num].reshape(28,28)
    im = Image.fromarray(tempimg)
    im.save(f"img/num{num}.jpg", "jpeg") 
    
########################################  function end



set_loaddata()
pre_dataprocess()
model, history =set_model()

#model 저장 
#model.save("model/mnist.h5")
#model load
#model= load_model('model/mnist.h5') #read model

evaluation(model, history)
predict_pro(model, range(100), True)
