#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : TESTS.py
# @Author: WangYe
# @Date  : 2018/8/1
# @Software: PyCharm
import numpy as np

a = np.array([[1,2],[3,4]])
sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)

print(sum0)
print(sum1)