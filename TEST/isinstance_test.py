#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : isinstance_test.py
# @Author: WangYe
# @Date  : 2018/8/21
# @Software: PyCharm
def f(liasd):

    result = isinstance(liasd, list)
    print(liasd,'instance of list?', result)

    # result = isinstance(numbers, dict)
    # print(numbers,'instance of dict?', result)
    #
    # result = isinstance(numbers, (dict, list))
    # print(numbers,'instance of dict or list?', result)
    #
    # number = 5
    #
    # result = isinstance(number, list)
    # print(number,'instance of list?', result)
    #
    # result = isinstance(number, int)
    # print(number,'instance of int?', result)
f([1,2,3])