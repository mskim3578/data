# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:06:55 2025

@author: letuin
"""

#seaborn 모듈 : 시각화모듈, 데이터셋
import pandas as pd
import seaborn as sns #시각화모듈

##################################################
# titanic 데이터셋 연습(데이터 전처리)
# seaborn 모듈에 저장된 데이터
'''
survived	생존여부
pclass	좌석등급 (숫자)
sex	성별 (male, female)
age	나이
sibsp	형제자매 + 배우자 인원수
parch: 	부모 + 자식 인원수
fare: 	요금
embarked	탑승 항구
class	좌석등급 (영문)
who	성별 (man, woman)
adult_male 성인남자여부 
deck	선실 고유 번호 가장 앞자리 알파벳
embark_town	탑승 항구 (영문)
alive	생존여부 (영문)
alone	혼자인지 여부
'''

#seaborn 모듈에 저장된 데이터셋 목록
print(sns.get_dataset_names())

#titanic데이터 로드. 
titanic = sns.load_dataset("titanic")
titanic.info()

'''
titanic.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
'''
titanic.head()

#pclass,class 데이터만 조회하기
titanic[["pclass","class"]].head()

#컬럼별 건수 조회하기
titanic.count() #결측값을 제외한 데이터
# 건수 중 가장 작은 값 조회하기
type(titanic.count())
titanic.count().min()
# 건수 중 가장 작은 값의 인덱스 조회하기
titanic.count().idxmin()


#titanic의 age,fare 컬럼만을 tidf 데이터셋에 저장하기
tidf = titanic[["age","fare"]]
tidf.info()

#tidf 데이터의 평균 데이터 조회
tidf.mean()

#tidf 데이터의 나이별 인원수를 조회. 최대 인원수를 가진 5개의 나이 조회
# value_counts sort됨
tidf.age.value_counts().head()

#tidf 데이터의 인원수가 많은 나이 10개 조회 
tidf.age.value_counts().head(10)

#tidf 데이터의 최대나이와 최소나이 조회
tidf.age.max()
tidf.age.min()

#승객 중 최고령자의 정보 조회하기
t1=tidf[tidf["age"]==tidf["age"].max()]
t1.unique() # dataframe 이여서 error
t2=tidf.iloc[tidf.age.idxmax()]
t2.unique()

# 데이터에서 생존건수(342), 사망건수(549) 조회하기
titanic["survived"].value_counts()
titanic["alive"].value_counts()


#성별로 생존건수 조회하기
cnt=titanic[["sex","survived"]].value_counts()
cnt.index

# 컬럼 : 변수,피처 용어사용.
# 상관 계수 : -1 ~ 1사이의 값. 변수의 상관관계 수치로 표현
# correlation coefficient
titanic.corr()  #문자 때문에 error
titanic.info()
titanic[["survived","age", "fare"]].corr()



#seaborn 데이터에서 mpg 데이터 로드하기
'''
mpg : 연비
cylinders : 실린더 수
displacement : 배기량
horsepower : 출력
weight : 차량무게
acceleration : 가속능력
model_year : 출시년도
origin : 제조국
name : 모델명
'''
mpg = sns.load_dataset("mpg")
mpg.info()

#  제조국별 자동차 건수 조회하기
mpg.origin.value_counts()
mpg["origin"].value_counts()

# 제조국 컬럼의 값의 종류를 조회하기. 
# unique() : 중복을 제거하여 조회 
t3=mpg.origin.unique()  #[usa, japan,europe]

# 출시년도의 데이터 조회하기
mpg.model_year.value_counts()
mpg.model_year.unique()

#  mpg 데이터의 통계정보 조회하기
mpg.describe()
mpg.describe()["cylinders"]


#mpg데이터의 행의값,열의값 조회
mpg.shape #(398,9)튜플데이터 : 398행 9열
#행의값 조회
mpg.shape[0]
#열의값 조회
mpg.shape[1]

#모든 컬럼의 자료형을 조회하기
mpg.dtypes

#mpg 컬럼의 자료형을 조회하기
mpg["mpg"].dtypes

# mpg. 데이터의 mpg,weight 컬럼의 최대값 조회하기
mpg.mpg.max()
mpg.weight.max()
mpg[["mpg","weight"]].max()
# mpg. 데이터의 mpg,weight 컬럼의 기술통계 정보 조회하기
mpg[["mpg","weight"]].describe()

# 최대 연비를 가진 자동차의 정보 조회하기
mpg[mpg["mpg"]==mpg["mpg"].max()]
mpg.loc[mpg["mpg"]==mpg["mpg"].max()]
mpg.iloc[mpg["mpg"].idxmax()]["mpg"]

#mpg 데이터의 컬럼간의 상관계수 조회하기
mpg.info()
mpg.columns
mpg.columns[:-2]

mpg.corr()
mpg[mpg.columns[:-2]].corr()  
#mpg mpg, weight 데이터의 컬럼간의 상관계수 조회하기
mpg[["mpg","weight"]].corr()


### 결측치 처리
titanic = sns.load_dataset("titanic")
titanic.info()
#deck   선실 고유 번호 가장 앞자리 알파벳
titanic.deck.unique()
#deck 컬럼의 값별 건수 출력하기
titanic.deck.value_counts() #결측값 제외한 값의 건수
#결측값을 포함한 값의 건수
titanic.deck.value_counts(dropna=False)
titanic.deck.head()
#isnull() : 결측값? 결측값인 경우 True, 일반값:False
titanic.deck.head().isnull()
#notnull() : 결측값아님? 결측값인 경우 False, 일반값:True
titanic.deck.head().notnull()



#결측값의 갯수 조회
titanic.isnull().sum() #컬럼별 결측값 갯수
titanic.isnull().sum(axis=0) #열(column)컬럼별 결측값 갯수
titanic.isnull().sum(axis=1) #행(index) 별 결측값 갯수
#결측값이 아닌 갯수 조회
titanic.notnull().sum()
titanic.notnull().sum(axis=0)
titanic.notnull().sum(axis=1)


########################
#dropna : 결측값 제거 
#         inplace=True 있어야 자체 변경 가능
# 해당하는 row를 지운다 


# NaN이 아닌 값(non-null 값)이 최소 thresh개는 있어야  이 행(또는 열)을 유지한다
# 
# axis=1, thresh=500  column의 결측값이 아닌 갯수가 thresh값 이상 있어야 한다 
# axis=0 index 결측값이 아닌 갯수가 thresh값 이상 있어야 한다

titanic = sns.load_dataset("titanic")
titanic.info()
titanic.info()
t4=titanic.dropna(axis=0)  # row에 null 이 있으면 삭제
t4.info()

df_tresh = titanic.dropna(axis=0,thresh=14)  #결측값이 아닌 갯수가 14개 이상
df_tresh.info()
df_tresh = titanic.dropna(axis=1,thresh=500)
df_tresh.info()





titanic.info()
#결측값을 가진 행을 제거
#subset=["age"] : 컬럼 설정.
#how='any'/'all' : 한개만결측값/모든값이 결측값
# axis=0 : 행
df_age = titanic.dropna(subset=["age"],how='any',axis=0)
df_age.info()


########################
# fillna : 결측값을 다른값으로 치환.
#          inplace=True가 있어야 자체 객체 변경

# fillna(치환할값,옵션)
#1. age 컬럼의 값이 결측값인 경우 평균 나이로 변경하기
#1. age 컬럼의 평균나이 조회하기
titanic["age"].value_counts(dropna=False)
titanic.info()
age_mean = titanic["age"].mean() 
age_mean

#치환하기
titanic["age"].fillna(age_mean,inplace=True)
titanic.info()


#2. embark_town 컬럼의 결측값은 빈도수가 가장 많은 
#   데이터로 치환하기
# embark_town 중 가장 건수가 많은 값을 조회하기
#value_counts() 함수 결과의 첫번째 인덱스값.-가장 큰수
embark_town=titanic["embark_town"]
most_freq=embark_town.value_counts().index[0]

#value_counts() 함수 결과의 가장 큰값의 인덱스값
most_freq = titanic["embark_town"].value_counts().idxmax()
most_freq


