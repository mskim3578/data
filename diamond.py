# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 15:10:51 2025

@author: DIJANG
"""

import random

# (5~15)*2 + 1 → 홀수 [11, 13, 15, ..., 31]
num = (random.randint(5, 15) * 2) + 1
print("num =", num)

point1 = num // 2
point2 = num // 2

for i in range(num):
    for j in range(num + 1):  # j는 0 ~ num
        if j == point1 or j == point2:
            print("*", end="")
        else:
            print(" ", end="")
    print()

    point1 -= 1
    point2 += 1

    if point1 == 0:
        point1 = num - 1
        point2 = 0
