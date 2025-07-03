import random
'''
ㅂ한자

'''
# 전역 변수
ladder = []  
result = []  

def start_connect(pos):
    for i in range(len(ladder[0])):
        if ladder[pos][i] == 1:
            pos += 1
        elif pos != 0 and ladder[pos - 1][i] == 1:
            pos -= 1
    return pos

def print_ladder():
    h1 = "           "
    h2 = "───────────"

    # 위쪽 번호에 의해 연결된 결과 프린트 
    for i in range(len(ladder)):
        result[start_connect(i)] = i
        print(f"{i+1}{h1}", end='')
    print()

    # 시작 수직선 출력
    for _ in range(len(ladder)):
        print("│" + h1, end='')
    print()

    # 가로줄(사다리) 출력
    for i in range(len(ladder[0])):
        for j in range(len(ladder)):
            if ladder[j][i] == 1:
                print("├" + h2, end='')
            elif j != 0 and ladder[j - 1][i] == 1:
                print("┤" + h1, end='')
            else:
                print("│" + h1, end='')
        print()

    # 마지막 수직선 출력
    for _ in range(len(ladder)):
        print("│" + h1, end='')
    print()

    # 결과 번호 출력
    for i in range(len(result)):
        print(f"{result[i]+1}{h1}", end='')
    print()

def run():
    global ladder, result

    vline = random.randint(4, 8)  
    # vline = 7  # 고정
    hline = 10

    ladder = [[0 for _ in range(hline)] for _ in range(vline)]
    result = [0 for _ in range(vline)]

    for _ in range(vline*4):
        f = random.randint(0, vline - 2)
        s = random.randint(0, hline - 2)
        if ladder[f + 1][s] == 0 and ladder[f - 1][s] == 0:
            ladder[f][s] = 1

    print_ladder()

if __name__ == "__main__":
    run()
