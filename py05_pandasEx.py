# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:05:09 2025

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

titanic = sns.load_dataset("titanic")
titanic.info()

'''
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


titanic.head(10)

# 1.pclass, class데이터만 head() 프린트 하기
titanic[['pclass', 'class']].head()

# 2 컬럼별 건수 조회하기
titanic.count()
type(titanic.count())
# 3. 건수 중 가장 작은 값 조회하기
titanic.count().min()

# 4. 건수 중 가장 작은 값의 인덱스 조회하기
titanic.count().idxmin()

# 5.titanic의 age,fare 컬럼만을 tidf 데이터셋에 저장하기
tidf=titanic[['age', 'fare']]
tidf.info()

# 6. tidf 데이터의 평균 데이터 조회
tidf.mean()

# 7. tidf 데이터의 나이 조회. 최대 나이를 가진 5개의 나이 조회
t1=tidf.sort_values(by='age', ascending=False)['age'].head()

# 8. tidf 데이터의 나이별 인원수를 조회. 최대 인원수를 가진 5개의 나이 조회
#  value_counts   sort descending
tidf.age.value_counts().head()
tidf.age.unique()


#9. tidf 데이터의 최대나이와 최소나이 조회
tidf.age.max()
tidf.age.min()


#10. 승객 중 최고령자의 정보 조회하기
tidf.age.idxmax()
titanic.iloc[tidf.age.idxmax()]

tidf.age.max()
titanic[titanic['age']==titanic.age.max()]

#11.데이터에서 생존건수, 사망건수 조회 하기
titanic['survived'].value_counts()
titanic['alive'].value_counts()

# 성별로 생존건수 조회하기
id1 = titanic[['sex', 'survived']].value_counts()
id1.index

# 컬럼 : 변수,피처 용어사용.
# 상관 계수 : -1 ~ 1사이의 값. 변수의 상관관계 수치로 표현
# correlation coefficient
titanic.corr()  #문자 때문에 error
titanic.info()
titanic[["survived","age", "fare"]].corr()

'''
          survived       age      fare
survived  1.000000 -0.077221  0.257307
age      -0.077221  1.000000  0.096067
fare      0.257307  0.096067  1.000000

'''

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

#1.  제조국별 자동차 건수 조회하기
mpg.origin.value_counts()
mpg["origin"].value_counts()

#2. 제조국 컬럼의 값의 종류를 조회하기. 
mpg["origin"].unique()

#3. 출시년도의 데이터 조회하기
mpg.model_year.value_counts()
mpg.model_year.unique()

#4.  mpg 데이터의 통계정보 조회하기
mpg.describe()

#5. mpg데이터의 행의값,열의값 조회
mpg.shape   #(398, 9)
mpg.shape[0]
mpg.shape[1]

#6. 모든 컬럼의 자료형을 조회하기
mpg.dtypes

#7. mpg 컬럼의 자료형을 조회하기
mpg["mpg"].dtypes

#8. mpg. 데이터의 mpg,weight 컬럼의 최대값 조회하기
mpg.mpg.max()
mpg.weight.max()
mpg[["mpg","weight"]].max()

#9. 최대 연비를 가진 자동차의 정보 조회하기
mpg[mpg['mpg']==mpg['mpg'].max()]
mpg.iloc[mpg['mpg'].idxmax()]


#10. mpg 데이터의 컬럼간의 상관계수 조회하기
mpg.info()
mpg.columns
c1=mpg[mpg.columns[:-2]]
c1.info()
mpg[mpg.columns[:-2]].corr()
mpg[["mpg","weight"]].corr()


### 결측치 처리
titanic = sns.load_dataset("titanic")
titanic.info()

#11. deck   선실 고유 번호 가장 앞자리 알파벳
titanic.deck.unique()
#12. deck 컬럼의 값별 건수 출력하기
titanic.deck.value_counts()
titanic.deck.value_counts(dropna=False)

#13. isnull() : 결측값? 결측값인 경우 True, 일반값:False
titanic.deck.head(10).isnull()

#14. notnull() : 결측값아님? 결측값인 경우 False, 일반값:True
titanic.deck.head().notnull()
titanic.info()
#15.  deck가 notnull()만 가진 dataframe을 만들어줘
#  df[조건식]
t_notnull=titanic[titanic.deck.notnull()]
t_null=titanic[titanic.deck.isnull()]



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

t4=titanic.dropna(axis=0)  # row에 null 이 있으면 삭제
t4=titanic.dropna(axis=1)  # column null 이 있으면 삭제
t4.info()

df_tresh = titanic.dropna(axis=0,thresh=14)  #결측값이 아닌 갯수가 14개 이상
df_tresh.info()
df_tresh = titanic.dropna(axis=1,thresh=200)
df_tresh.info()


titanic.info()
#결측값을 가진 행을 제거
#subset=["age"] : 컬럼 설정.
#how='any'/'all' : 한개만결측값/모든값이 결측값
# subset=["age"]값이 null이 아닌 row는 삭제 하지 않는다
# axis=0 : 행
df_age = titanic.dropna(subset=["age"],how='any',axis=0)
df_age.head(10)

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


titanic["age"].fillna(age_mean, inplace=True)

#2. embark_town 컬럼의 결측값은 빈도수가 가장 많은 
#   데이터로 치환하기
# embark_town 중 가장 건수가 많은 값을 조회하기
#value_counts() 함수 결과의 첫번째 인덱스값.-가장 큰수

t3=titanic["embark_town"].value_counts()
t3

most_freq=titanic["embark_town"].value_counts().index[0]

titanic["embark_town"].fillna(most_freq, inplace=True)

# embarked 컬럼을 앞의 값으로 치환하기
# embarked 컬럼의 값이 결측값인 레코드 조회하기
#앞의 데이터로 치환하기
# method="ffill" : 앞의 데이터로 치환
# method="bfill" : 뒤의 데이터로 치환
# method="backfill" : 뒤의 데이터로 치환

titanic = sns.load_dataset("titanic")

t_null=titanic[titanic.embark_town.isnull()]
titanic[58:65]["embark_town"]
'''
58    Southampton
59    Southampton
60      Cherbourg
61            NaN
62    Southampton
63    Southampton
64      Cherbourg
'''
titanic[825:834]["embark_town"]
'''

825     Queenstown
826    Southampton
827      Cherbourg
828     Queenstown
829            NaN
830      Cherbourg
831    Southampton
832      Cherbourg
833    Southampton
'''

titanic.embark_town.fillna(method='backfill', inplace=True)
titanic[58:65]["embark_town"]
titanic[825:834]["embark_town"]


############  
mpg = pd.read_csv("data/auto-mpg.csv")
mpg.info()

#kpl : kilometer per liter mpg * 0.425
mpg["kpl"]=mpg["mpg"]*0.425
mpg.info()
mpg.kpl.head()

# round(1) : 소숫점 한자리로 반올림
mpg["kpl"]=mpg["kpl"].round(1)












