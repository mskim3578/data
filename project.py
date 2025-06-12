# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:11:12 2025

@author: EMS
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Malgun Gothic")





df = pd.read_excel("data/남북한발전전력량.xlsx")
df.info()
df
#북한지역의 발전량만 조회.
df = df.loc[5:]
df
df.info()








#전력량 (억kwh) 컬럼 제거  ======!!!!    억kwh 특수문자를 수정하였음
#df.drop("전력량 (억kwh)",axis=1, inplace=True) 
del df["전력량 (억kwh)"]
df.info()
df["발전 전력별"]
#발전 전력별 컬럼을 index로 설정
df.set_index("발전 전력별",inplace=True)
df.info()
df.index
df

#전치 행렬
df=df.T





#1)  합계 컬럼을 총발전량 컬럼으로 변경하기
df=df.rename(columns={'합계':'총발전량'})

#총발전량-1년 추가 :전년도 발전량
df.head()
#shift(1) : 총발전량의 앞의 인덱스 데이터
df["전년도발전량"] = df["총발전량"].shift(1)
df.head()

#증감율 컬럼 추가하기
# 증감율 :(현재-전년도)/전년도 * 100
#         (현재/전년도 - 1) * 100
df["증감율"]=((df["총발전량"]/df["전년도발전량"]) - 1) * 100

plt.plot(df.index, df['수력'], label='수력')
plt.plot(df.index, df['화력'], label='화력')
plt.xticks(rotation=45)
plt.legend()
plt.show()
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Malgun Gothic")

df[['수력','화력']].plot(kind='line')
plt.show()

df[['수력','화력']].plot(kind='bar')
plt.savefig("data/북한전력량.png", dpi=400, bbox_inches="tight")
plt.show()

#자동차 연비데이터의 mpg 값을 히스토그램으로 출력하기
mpg = sns.load_dataset("mpg")
mpg.info()

mpg['mpg'].plot(kind='hist')
plt.show()

df["mpg"].plot(kind="hist",bins=20,color='coral',\
               figsize=(10,5),histtype='bar',linewidth=1)
plt.title("MPG 히스토그램")
plt.xlabel("mpg(연비)")
plt.show()


'''
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   mpg           398 non-null    float64
 1   cylinders     398 non-null    int64  
 2   displacement  398 non-null    float64
 3   horsepower    392 non-null    float64
 4   weight        398 non-null    int64  
 5   acceleration  398 non-null    float64
 6   model_year    398 non-null    int64  
 7   origin        398 non-null    object 
 8   name          398 non-null    object 
 '''
