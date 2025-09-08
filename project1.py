

import openpyxl 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#  렛유인 project1 
plt.rcParams['axes.unicode_minus'] = False  #chart에 마이너스 프린트 
plt.rc("font",family="Malgun Gothic")   # 한글 프린트   

def set_loaddata(filename=''):
    global all_sheets, sheetname, fname
    if not filename : filename = "data1/Raw data_ref_C_EX1.xlsx"
    all_sheets = pd.read_excel(filename, sheet_name=None, header=None)
    all_sheets.keys()  # dictionary
    len(all_sheets.keys())
    sheetname = list(all_sheets.keys())
    fname=filename



     
def pre_process(sheetindex):
    global number_df
    
    ### 1. 엑셀 파일에 필요한 sheet 선택한다 
    df = all_sheets[sheetname[sheetindex]]   # 0 -> Yield, 1 -> ET(DC)
    
    ### 2. df.columns  setting
    # temp1 = df.iloc[0].fillna(method="ffill").to_list()  # deprecate 됬음 
    # df.ffill() nan 일때 앞의 값으로 채운다 
    # df.bfill() nan 일때 뒤의 값으로 채운다 
    # df.fillna('') nan 일때 '' 로 채운다 
    
    # col1 에서 null일 때  앞의 값으로 채운다 
    col1 = df.iloc[0].ffill().fillna('').to_list()
    col2 = df.iloc[1].ffill().fillna('').to_list() 
 
    # c1, c2 중에 ''이 있으면 '_'을 넣지 않는다 
    df.columns=[c1 + c2  if c1=='' or c2 == '' else c1 + "_"+ c2 for c1, c2 in zip(col1, col2)]
    # print(sheetname[sheetindex], df.columns)
    
    ### 3. 처음에 두개의 columns으로 사용한 row를 삭제한다
    df=df.drop(index=[0,1]) # df.iloc[0], df.iloc[1]  삭제
    
    ### 4. 숫자만 가지고 있는 정보를 확인 한다 
    number_col=df.columns[3:].to_list()  #No.	LOTID	WFID columns에서 제거한다 
    number_df=df[number_col]
    
    number_df=number_df.convert_dtypes()	# 자료마다 type변경을 한다
    
    # 숫자 타입 컬럼만 선택하여 새로운 DataFrame 생성
    # include='number'는 정수(int), 실수(float) 등 모든 숫자 타입을 포함합니다.
    number_df = number_df.select_dtypes(include='number') # number type만 었는다
    
    number_df.info()   # null을 확인 한다 
   

def all_graph(n_df, allview=True, sigma=2, graph_type='plot', figurename=None ):
  
  plt.figure(figsize=(10, 10), num=figurename)
  count=1
  row_count = len(n_df.columns)//6 + 1 #몫 더하기 1을 한다  
  labelname1 = "sigma-"+str(sigma)
  labelname2 = "sigma+"+str(sigma)  
  plt.suptitle(fname+" : "+sheetname[sheetindex])
  
  for col in n_df.columns :
    
    xx= n_df[col].values
    df_mean = n_df[col].mean()
    df_std = n_df[col].std()
    
    
   
    #print(col, df_std, df_mean)
    x = range(len(xx)) 
    
    upperdf= n_df[n_df[col]>(df_mean + (df_std*sigma))]
    lowerdf= n_df[n_df[col]<(df_mean - (df_std*sigma))]
   
    
    
    # df[col].fillna(df_mean) # 확인한다 !!!!!!!!!
    
    # sigma - 2,3 보다 큰 자료가 있거나, allview 가 True 일때 
    if len(upperdf) != 0 or len(lowerdf) != 0 or allview:
      ax = plt.subplot(row_count, 6, count)
      
      count +=1
      if graph_type == 'hist':
        plt.hist(xx)
      elif graph_type == 'scatter':
        plt.plot(x, [df_mean - (sigma * df_std) for x in range(len(xx))], label=labelname1)   
        plt.plot(x, [df_mean + (sigma * df_std) for x in range(len(xx))], label=labelname2)
        plt.scatter(x, xx)
      else:
        plt.plot(x, [df_mean - (sigma * df_std) for x in range(len(xx))], label=labelname1)   
        plt.plot(x, [df_mean + (sigma * df_std) for x in range(len(xx))], label=labelname2)
        plt.plot(x,xx,  color='b', linestyle='-')  
       
      #plt.legend()  
     
      plt.title(f"{col}", fontsize=8)
      plt.axis("off")  #x, y 좌표 프린트 않한다   
      
    
  plt.tight_layout()  #graph 자동 조정
  plt.show(block=True)         
    
###########################   function end

excel_num="2"   # data1/Raw data_ref_C_EX2.xlsx
set_loaddata("data1/Raw data_ref_C_EX"+excel_num+".xlsx")  #한개의 excel file을 읽어서 sheet별로 작업한다
sheetindex=1 #sheet의 순서에 따른 index
pre_process(sheetindex)  # 전처리  set column

number_df.info()
all_graph(number_df, sigma=2, allview=False, figurename=sheetname[sheetindex] ) 
  
