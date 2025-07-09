# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 21:24:18 2025

@author: DIJANG
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:55:14 2023

@author: family
"""

#############################
#  이항분류 : 분류의 종류가 2종류인 경우
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as plt

def set_dataload():
    
    global wine, wine_norm

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    red = pd.read_csv(url+'winequality-red.csv', sep=';') #red 와인 정보
    white = pd.read_csv(url+'winequality-white.csv', sep=';') #white 와인 정보
    red.info() #1599
    white.info() #4898
    '''
    1 - fixed acidity : 주석산농도
    2 - volatile acidity : 아세트산농도
    3 - citric acid : 구연산농도
    4 - residual sugar : 잔류당분농도
    5 - chlorides : 염화나트륨농도
    6 - free sulfur dioxide : 유리 아황산 농도
    7 - total sulfur dioxide : 총 아황산 농도
    8 - density : 밀도
    9 - pH : ph
    10 - sulphates : 황산칼륨 농도
    11 - alcohol : 알코올 도수
    12 - quality (score between 0 and 10) : 와인등급
    '''
    #type 컬럼 추가
    #red와인인경우 type컬럼에 0, white와인인 경우 type컬럼에 1 을 저장하기.
    red["type"]=0
    white["type"]=1
    #red,white 데이터를 합하여 wine 데이터에 저장하기
    wine = pd.concat([red,white])
    wine.to_csv("wine.csv")
    wine.info()
    wine.head()
    
    #wine 데이터를 minmax 정규화하여 wine_norm 데이터에 저장
    wine.min()
    wine.max()
    wine_norm = (wine-wine.min()) / (wine.max()-wine.min())
    wine_norm.head()
    wine_norm.min()
    wine_norm.max()
    wine_norm["type"].head()
    wine_norm["type"].tail()


def set_preprocess():
    global train_x,train_y,test_x,test_y
    # wine_norm 데이터를 섞어 wine_shuffle 데이터에 저장하기.
    #sample() : 임의로 표본추출을 위한 함수. 
    #frac=1 : 표본추출의 비율. 1은 100%. 
    wine_shuffle = wine_norm.sample(frac=1)
    wine_shuffle["type"].head(10)
    wine_shuffle["type"].tail(10)
    wine_shuffle.info()
    #wine_shuffle 데이터를 배열데이터 wine_np로 저장
    wine_np = wine_shuffle.to_numpy()
    type(wine_np)
    wine_np.shape    
    #train(8),test(0) 데이터 분리.
    #설명변수,목표변수(정답)로 분리
    train_idx = int(len(wine_np)*0.8)
    train_idx
    #훈련데이터 분리
    train_x,train_y = \
        wine_np[:train_idx,:-1],wine_np[:train_idx,-1]
    train_x.shape    #(5197, 12)
    train_y.shape    #(5197,)
    #테스트데이터 분리
    test_x,test_y = \
        wine_np[train_idx:,:-1],wine_np[train_idx:,-1]
    test_x.shape #(1300, 12)
    test_y.shape #(1300,)
    #LABEL을 onehot 인코딩하기

    train_y = tf.keras.utils.to_categorical(train_y,num_classes=2)
    test_y = tf.keras.utils.to_categorical(test_y,num_classes=2)
   

def set_model():
    global model, history
    #모델 생성
    model = Sequential([
        Dense(units=12,activation='relu',input_shape=(12,)),
        Dense(units=2,activation='sigmoid')    #이중분류 사용. 
        ])
    model.summary()
    #binary_crossentropy : 이중분류에서 사용되는 손실함수
    #                      레이블을 onehot 인코딩 필요
    model.compile(optimizer="adam", loss='binary_crossentropy',\
                  metrics=['accuracy'])
    #validation_split=0.25 : 25%의 데이터를 검증데이터로 사용    
    history = model.fit(train_x,train_y,epochs=25,batch_size=32,\
                        validation_split=0.25)
        
def get_evaluation():
    #학습결과 시각화하기.
    # 학습데이터와 검증 데이터의 loss,accuracy 값을 선그래프로 출력하기
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    #'b-' : blue 실선
    plt.plot(history.history['loss'], 'b-', label='loss')
    #'r--' : red, 점선
    plt.plot(history.history['val_loss'], 'r--', label='val_loss') 
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'b-', label='accuracy') 
    plt.plot(history.history['val_accuracy'], 'r--',\
             label='val_accuracy')
    plt.xlabel('Epoch') 
    plt.ylim(0.7, 1) 
    plt.legend()
    plt.show()  
    # 과적합 발생 안됨.
    #평가하기
    model.evaluate(test_x,test_y) #[0.0324331559240818, 0.9930769205093384]
    #예측하기
    results = model.predict(test_x)
    np.argmax(results[:10],axis=-1)
    np.argmax(test_y[:10],axis=-1)
    #평가 결과 출력하기 : 혼동행렬, heatmap 출력하기 
    #혼동 행렬(confusion_matrix)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.argmax(test_y,axis=-1),\
                          np.argmax(results,axis=-1))
    cm  
    import seaborn as sns
    plt.figure(figsize = (7, 7))
    sns.heatmap(cm, annot = True, fmt = 'd',cmap = 'Blues')
    plt.xlabel('predicted label', fontsize = 15)
    plt.ylabel('true label', fontsize = 15)
    plt.xticks(range(2),['red','white'],rotation=45)
    plt.yticks(range(2),['red','white'],rotation=0)
    plt.show()

# from sklearn.metrics import classification_report
# classification_report\
#   (np.argmax(test_y, axis = -1), np.argmax(results, axis = -1))


def get_prediction():
    
    # 한개의 자료(전처리 전 자료) 예측 문제 
    # 원자료  
    index=5000
    wine.iloc[index]
    checkWine(300)
    class_names=['red', "white"]


    test_wine_np=((wine-wine.min())/(wine.max()-wine.min())).to_numpy()
    test_wine_x=test_wine_np[:, :-1]
    test_wine_y=test_wine_np[:, -1]
    results=model.predict(test_wine_x) 
    results[100:111]
    test_wine_y[100:111]
    truecount=0
    for idx in range(len(wine)) :
        pred_y = int(np.argmax(results[idx]))
        label_y = int(test_wine_y[idx])
        if pred_y == label_y:
            truecount +=1
        else:
            # print(f'label:{label_y}--> pred:{pred_y}')
           print(f'{idx}->label:{class_names[label_y]}--> pred:{class_names[pred_y]}')
        
    
    print('real accuracy:',truecount/len(wine)*100)  #6449

def checkWine(index):
    #1 pandas to numpy 
    test_wine_np = wine.iloc[index].to_numpy() #win_norm에 index row
    #2 minmax정규화
    test_wine_np=((test_wine_np-wine.min())/(wine.max()-wine.min())).to_numpy()  #정규화
    test_wine_np
    
    #3 독립변수 x 
    test_wine_x=test_wine_np[:-1]
    test_wine_x
    #4 test_x (1300, 12)  #shape setting
    test_wine_x=test_wine_x.reshape((1,12)) 
    #예측
    test_wine_pre=model.predict(test_wine_x) 
   # print(np.argmax(test_wine_pre),":",test_wine_np[-1]) #     수정 하였음
    print(test_wine_np[-1],":",np.argmax(test_wine_pre)) 
########################   function end


set_dataload()
set_preprocess()
set_model()
get_evaluation()
get_prediction()

# 한개의 row자료 확인
checkWine(16)










