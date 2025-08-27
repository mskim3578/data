# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 14:44:02 2025

@author: letuin
"""

####  지정 지역의 인구분포와 가장 비슷한 지역찾기
# 1세 ~ 10세  age.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# thousands="," 천의자리 콤마임, 
# index_col=0   0번 column 번호를 index로 한다 , 
# header=None 컬럼적용 않함 
df = pd.read_csv("data/age.csv",encoding="cp949", thousands=",", index_col=None,header=None)
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3911 entries, 0 to 3910
Columns: 104 entries, 0 to 103
dtypes: object(104)
memory usage: 3.1+ MB
'''
df = pd.read_csv("data/age.csv",encoding="cp949", thousands=",", index_col=0)
'''
<class 'pandas.core.frame.DataFrame'>
Index: 3910 entries, 서울특별시  (1100000000) to 제주특별자치도 서귀포시 예래동(5013062000)
Columns: 103 entries, 2025년06월_계_총인구수 to 2025년06월_계_100세 이상
dtypes: int64(103)
memory usage: 3.1+ MB
'''

df.info()
df.columns
df.index
type(df.index)

name1=df.index.str.contains("역삼1동")  # 조건식
name2=list(map(lambda x : not x  , name1))

# df[조건식] 조건식이 True인것을 찾아서 df2를 만든다

df2=df[name1]  # 기준 지역이  역삼 1동
df2.info()
df3=df[name2]  # 역삼1동 제외


home1 = df.iloc[0]  # 한개의 row (index)

home2= df.iloc[0][1:]  # 연령별 인구수
home3= df.iloc[0][0] # 총인구수


home=df2.iloc[0][1:]/df2.iloc[0][0] # 역삼 1동의 연령별 인구수/ 총인구수



tdf4=df3.T
mn=1

for label, content in df3.T.items():
    #print(label, content)
    away=content[1:]/content[0]  #  0 ~ 10 / 전체 11개
    s=sum((home-away) ** 2)
    if s < mn :
        mn = s
        result=content[2:]
        name=result.name
        print(name)
 
        
print(name)
#   서울특별시 강남구 논현1동(1168052100)
name.find("(")
name=name[:14]
col_name=df.columns[2:].tolist()
col_name=[x.split("_")[2] for x in col_name]

#


 

plt.figure(figsize = (10,5), dpi=100)      
plt.title('역삼1동 지역과 가장 비슷한 인구 구조를 가진 지역')
plt.plot(df2.iloc[0][2:], label="역삼1동")
plt.plot(result, label=name)
plt.xticks(range(len(df2.iloc[0][2:])), col_name,rotation=45,fontsize="small")
plt.legend()
plt.show()



plt.figure(figsize = (10,5), dpi=100)      
plt.title('역삼1동 지역과 가장 비슷한 인구 구조를 가진 지역')
plt.plot(df2.iloc[0][2:], label="역삼1동")
###  df 자료중 5번째 (df.iloc[4]) 자료와 역삼 1동 비교 그래프
plt.plot(df.iloc[4][2:], label=df.index[4])
plt.xticks(range(len(df2.iloc[0][2:])), col_name,rotation=45,fontsize="small")
plt.legend()
plt.show()


name.find("(")
name=name[:14] # 서울특별시 서대문구 신촌동

plt.figure(figsize = (10,5), dpi=100)      
plt.title('역삼1동 지역과 가장 비슷한 인구 구조를 가진 지역')
plt.plot(df2.iloc[0][1:], label = df2.iloc[0].name)
plt.plot(result, label=result.name)

# serise일 len(result) , value 가 y

plt.xticks(range(len(df2.iloc[0][1:])), col_name[1:],rotation=45,fontsize="small")
plt.legend()
plt.show()   

##########  8.26일

import pandas as pd
import matplotlib.pyplot as plt


###  연합 그래프 작성
df = pd.read_excel("data/남북한발전전력량.xlsx")   # github data에 있음
df.info()
df

### 1. 북한의 발전양만 선택한다
df=df.iloc[5:]  # 5 ~ 8까지
 
### 2. 전력량 (억kwh) 삭제 한다
del df['전력량 (억kwh)']

####3. '발전 전력별'를 index로 정한다
df.set_index('발전 전력별', inplace=True)


####4. index와 columns 바꾼다
df=df.T 

### 5.'합계' "총발전량" column name을 수정한다
df = df.rename(columns={'합계':'총발전량'})


### 6. 총발전량-1년 추가 :전년도 발전량

df["전년도발전량"]=df['총발전량'].shift()
df.head()

#  7. 증감율 컬럼 추가하기
# 증감율 : (현재-전년도)/전년도 * 100
#         (현재/전년도 - 1) * 100
df["증감율"]=(df['총발전량']/df['전년도발전량'] -1) * 100

'''
발전 전력별 총발전량   수력   화력 원자력 전년도발전량        증감율
1990    277  156  121   -   None        NaN
1991    263  150  113   -    277  -5.054152
1992    247  142  105   -    263   -6.08365
1993    221  133   88   -    247 -10.526316
1994    231  138   93   -    221   4.524887
'''


#ax1 = df[['수력']].plot(kind='bar', figsize=(20, 10), width=0.7) 
x=df.index
y1=df['수력']
y2=df['증감율']

plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots()
ax1.bar(x, y1)
ax1.set_ylabel('ax1', color='b')
#ax1.set_ylim(0, 160) 
ax1.set_xticks(x)
ax1.set_xticklabels(x, rotation=45)  # <- 여기서 회전 설정!
ax2=ax1.twinx()

ax2.plot(x, y2, color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylabel('ax2', color='r')
#ax2.set_ylim(0, 1.75) 

plt.title("Average Yield per HDP_DEPO (EQPID + Chamber)")

plt.tight_layout()
plt.show()



# fig, ax1 = plt.subplots()
# ax1.bar(df.index, df.수력) 

ax1 = df[['수력','화력']].plot(kind='bar', figsize=(20, 10), width=0.7) 
ax2 = ax1.twinx()
ax2.plot(df.index, df.증감율, ls='--', marker='o', markersize=10, 
        color='green', label='전년대비 증감율(%)')
ax1.set_ylim(0, 200) #막대그래프 y축의 값의 범위
ax2.set_ylim(-50, 50)#선그래프  y축의 값의 범위  
ax1.set_xlabel('연도', size=20) #x축값의 설명
ax1.set_ylabel('발전량(억 KWh)') #막대그래프 y축값의 설명
ax2.set_ylabel('전년 대비 증감율(%)') #선그래프 y축값의 설명
plt.title('북한 전력 발전량 (1990 ~ 2016)', size=30) #전체 그래프 제목
ax1.legend(loc='upper left') #범례: 왼쪽 위 위치
ax2.legend(loc='upper right') #범례: 오른쪽 위 위치
#그래프를 이미지파일로 저장.
#savefig(저장파일명,해상도,이미지크기설정)
#plt.savefig("img/북한전력량.png", dpi=400, bbox_inches="tight")
plt.show() #화면에 표시.

#######   dataframe.plot(.......)
#자동차 연비데이터의 mpg 값을 히스토그램으로 출력하기
df = sns.load_dataset("mpg")
df.info()


#1. DataFrame plot 히스토그램 출력
df['mpg'].plot(kind='hist', bins=20, color='coral',\
               figsize=(10,5),histtype='bar', width=1.5)
plt.title("MPG 히스토그램")
plt.xlabel("mpg(연비)")    
plt.show()


# weight,mpg 파이 그래프 데이터의 산점도 출력하기
# df.plot(kind='scatter', x="weight", y='mpg', 
#         color='coral', figsize=(10,5),s=10)
# cmap='magma',c=df["cylinders"]  cylinders의 값에 따라 점의 color(cmap) 정해진다
df.plot(kind="scatter",x="weight",y="mpg",marker="+", figsize=(10,5),\
         cmap='magma',c=df["cylinders"],s=50,alpha=0.7)   

plt.title("산점도:mpg-weight-cylinders")
#plt.scatter(df["weight"], df['mpg'],  color='coral',s=10)
plt.show()

# origin을 이용해 파이 그래프를 그린다
origin_df = df.origin.value_counts()

origin_df.plot(kind='pie', figsize=(7,5),autopct="%.1f%%",startangle=90,
                  colors=['chocolate','bisque','cadetblue'])
plt.title('자동차 생산국')
plt.legend(labels=origin_df.index, loc="upper left")
plt.show()

#########  boxplot 
usa_df=df[df['origin']=='usa']["mpg"]
japan_df=df[df['origin']=='japan']["mpg"]
eu_df=df[df['origin']=='europe']["mpg"]


data=[usa_df, japan_df, eu_df]
plt.boxplot(data, labels=['usa', 'japan', 'europe'], patch_artist=True)
plt.title("제조국자별 연비분포(수직박스플롯)")
plt.show()
data=[usa_df, japan_df, eu_df]
plt.boxplot(data, labels=['usa', 'japan', 'europe'], 
            patch_artist=True, vert=False)
plt.title("제조국자별 연비분포(수평박스플롯)")
plt.show()


# 박스그래프: 두개의 그래프 출력하기
fig = plt.figure(figsize=(15,5)) #그래프출력영역,크기지정.
#그래프 출력영역을 분리
ax1 = fig.add_subplot(1,2,1) #1행2열 첫번째 그래프
ax2 = fig.add_subplot(1,2,2) #1행2열 두번째 그래프

ax1.boxplot(data, labels=['usa', 'japan', 'europe'], patch_artist=True)
ax2.boxplot(data, labels=['usa', 'japan', 'europe'], 
            patch_artist=True, vert=False)

ax1.set_title("제조국자별 연비분포(수직박스플롯)")
ax2.set_title("제조국자별 연비분포(수평박스플롯)")
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].boxplot(data, labels=['usa', 'japan', 'europe'], patch_artist=True)
ax[0].set_title("제조국자별 연비분포(수직박스플롯)")

ax[1].boxplot(data, labels=['usa', 'japan', 'europe'], 
            patch_artist=True, vert=False)
ax[1].set_title("제조국자별 연비분포(수평박스플롯)")
plt.tight_layout()
plt.show()

#  fit_reg=False  회기선 프린트 여부
titanic = sns.load_dataset("titanic")
titanic.info()
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 2, 1) 
ax2 = fig.add_subplot(1, 2, 2)
sns.regplot(x='age',  y='fare', data=titanic, ax=ax1)  
sns.regplot(x='age', y='fare',  data=titanic, ax=ax2, fit_reg=False)  
plt.show()


fare_summary = titanic.pivot_table(index="sex", columns="pclass",
                                   values='fare', aggfunc="size")
# count : null 포함하지 않음
# size : null 포함
titanic["sex"].unique()
titanic["pclass"].unique()
'''
pclass           1          2          3
sex                                     
female  106.125798  21.970121  16.118810
male     67.226127  19.741782  12.661633


'''

sns.heatmap(fare_summary,annot=True,fmt='d',
            cmap='Reds',linewidth=.5,cbar=True)
plt.show()




### boxplot 그래프
fig = plt.figure(figsize=(15, 10))   
ax1 = fig.add_subplot(2, 2, 1) 
ax2 = fig.add_subplot(2, 2, 2) 
ax3 = fig.add_subplot(2, 2, 3) 
ax4 = fig.add_subplot(2, 2, 4) 
'''
data=titanic : 데이터변수명.
x="alive", y="age" : titanic의 컬럼명.
hue='sex' : 성별로 분리(unique()).

violinplot : 값의범주+분포도를 표시. 가로길이 넓은부분은 분포가 많은 수치의미
'''
sns.boxplot(x='alive', y='age', data=titanic, ax=ax1) 
sns.boxplot(x='alive', y='age', hue='sex', data=titanic, ax=ax2) 
sns.violinplot(x='alive', y='age', data=titanic, ax=ax3) 
sns.violinplot(x='alive', y='age', hue='sex', data=titanic,ax=ax4) 
ax2.legend(loc="upper center")
ax4.legend(loc="upper center")
plt.show()


#pairplot : 각각의 컬럼들의 
#           산점도출력. 대각선위치는 히스토그램으로 표시.
#           값의 분포, 컬럼간의 관계.
titanic_pair = titanic[["age","pclass","fare"]]    
titanic_pair
sns.pairplot(titanic_pair)
plt.show()




import pandas as pd
chipo = pd.read_csv("data/chipotle.tsv",sep="\t")
chipo.info()

'''
데이터 속성 설명
order_id : 주문번호
quantity : 아이템의 주문수량
item_name : 주문한 아이템의 이름
choice_description : 주문한 아이템의 상세 선택 옵션
item_price : 주문 아이템의 가격 정보
'''


#1. chipo 데이터의 행열의 갯수 출력하기
chipo.shape

#2.컬럼명
chipo.columns

#3. 인덱스명
chipo.index

#4. orderid : str,  item_price : float  타입 변경
'''
#   Column              Non-Null Count  Dtype 
---  ------              --------------  ----- 
 0   order_id            4622 non-null   int64 
 1   quantity            4622 non-null   int64 
 2   item_name           4622 non-null   object
 3   choice_description  3376 non-null   object
 4   item_price          4622 non-null   object
'''

chipo.order_id = chipo.order_id.astype(str)
chipo.info()
type(chipo.item_price)
chipo.item_price = chipo.item_price.str.replace("$","").astype(float)

#5.  상품명을 조회 , 상품의 갯수

chipo.item_name.unique()
len(chipo.item_name.unique())


#6. 주문금액 합계
hap=chipo.item_price.sum()

#7 주문건수
chipo.order_id.unique()
cnt=len(chipo.order_id.unique())

#8 주문당 평균 금액
hap/cnt
g1=chipo.groupby("order_id")
chipo.groupby("order_id")['item_price'].sum().mean()

#9. 50달러 이상 주문한 주문번호를 조회
g2=chipo.groupby("order_id").sum()
result=g2[g2['item_price']>=50]
list(result.index)
result.index

#10. 50달러 이상 주문한 주문정보를 조회

chipo_50 = chipo[chipo.order_id.isin(result.index)]
chipo_50.shape

chipo_51 = chipo.groupby("order_id")\
        .filter(lambda x : sum(x["item_price"]) >=50 )

chipo_51.info()


#11. item_name별 단가 (min)

price_one= chipo.groupby("item_name").min().to_dict()  
#group(min,max...)  함수가 있어야 dictionary로 수정된다 

price_one= chipo.groupby("item_name")

for key, item in price_one:
    print(key, len(item), type(item))


len(chipo["item_name"].unique())

g4=chipo[chipo["item_name"]=='Chicken Bowl']


price_min = chipo.groupby("item_name").min()["item_price"]
price_min.max()

plt.rc("font", family="Malgun Gothic")
plt.hist(price_min, bins=10)
plt.xticks(range(10))
plt.ylabel("상품갯수")
plt.title("상품단가 분포")
plt.show()

price_min.plot(kind = "hist", bins=10)
plt.xticks(range(10))
plt.ylabel("상품갯수")
plt.title("상품단가 분포")
plt.show()


#12. 주문당 금액이 가장 높은 5건의 주문 총수량을 조회하기

chipo.groupby("order_id").sum()\
            .sort_values(by='item_price', ascending=False)[:5]


#13.  Chicken Bowl 몇번 주문되었는지 출력하기
chip_chicken = chipo[chipo['item_name']=='Chicken Bowl'] #729

len(chip_chicken.groupby("order_id"))  #615
len(chip_chicken.drop_duplicates(['item_name', 'order_id'])) #615


