# -*- coding: utf-8 -*-
"""
global 함수는 파라메터 name으로 사용 하면 않된다 

문제)
sqsize=10 10인 정사각형에 random하게 점을 찍어서
시계방향, 반시계방향으로 돌리고 흩허진 점을을 아래로 떨어뜨리는 
프로그램을 작성 하세요



"""

import numpy as np



def rotation_box(trun):
    global sandbox
    if trun == turn_left:
        sandbox=np.rot90(sandbox, k=-1)     
    else:
        sandbox = np.rot90(sandbox)  
    #print(sandbox)
    print_box()

def drop_down():
    global sandbox
    for x in range(sqsize):
        countx = 0
        for y in range(sqsize):
            if sandbox[y][x] == 1 : countx +=1
        # print(countx)
        for y in range(sqsize):
            if y<(sqsize-countx) :  # (5-0)
              sandbox[y][x]  = 0
            else:
              sandbox[y][x]  = 1 
    '''
    1이 3개이면 0은 2(len-3)개임
    단 0은 y<3이여야 한다
    
    '''
    print_box()

def random_reset():
    global sandbox 
    sandbox = np.random.randint(0, 2, size=(sqsize, sqsize))
    print_box()

def run():
    global sqsize, turn_left, turn_right
    
    turn_left=0  #반시계
    turn_right=1  #시계
    sqsize=10
    
    
    
    
    random_reset()
    while True:
        try:
            value = int(input("반시계 방향 회전은 1, 시계방향 회전은 2, 블럭 떨어뜨리기 3, random reset 4, 종료는 9:> "))
            if value in [1, 2, 3, 4, 9]:
                if value == 1:
                    rotation_box(turn_right)
                elif value == 2:
                    rotation_box(turn_left)
                elif value == 3:
                    drop_down()
                elif value == 4:
                    random_reset()
                else:
                    break
                
            else:
                print("1, 2, 3, 4, 9 중에서 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
            
def print_box():
   
   # ┌  ㅂ한자 , ㅁ한자  
   print("┌", end="")
   print("──"*sqsize, end="")
   print("┐", end="\n")  
   
   for i in range(sqsize):
       #print(i)
       print("│", end="")  
       for j in range(sqsize):
           if sandbox[i][j] == 1:
               print("● " ,end="")
           else:
               print("  ", end="")
       print("│", end="\n") 
       
  
   print("└", end="")
   print("──"*sqsize, end="")
   print("┘", end="\n")  
   

run()

