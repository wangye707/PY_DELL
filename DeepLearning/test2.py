#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : test2.py
# @Author: WangYe
# @Date  : 2018/8/1
# @Software: PyCharm

def f1(**kwargs):
    allowed_kwargs = {
        "input_shape",
        "input_dim"
    }
    if kwargs.keys() not in list(allowed_kwargs):
        return
    else:
        print(kwargs)


f1(input_shape=32)