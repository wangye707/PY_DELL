#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : Test.py
# @Author: WangYe
# @Date  : 2019/1/9
# @Software: PyCharm

# N = input()
#
# num = input()


#num1 = num.split(" ")  #list形式的输入

import copy

N = int(input())

num = input()
num = num.split(" ")

#print(type(num))
num1 = copy.deepcopy(num)

last_list = []

#print(type(num1))

last_list_len = []
for i in range(int(N)):

    temp = int(num1[i])

    temp_every_len = 0

    #print(temp)
    while temp > 1:     #取最高位数，7 13 4 246的最高位数是3

        temp = temp /10
        #print("aa",temp)
        temp_every_len += 1

    #print("位数",temp_every_len)
    last_list_len.append(temp_every_len)

    now_num = int(num1[i])  # 当前计算数字
    temp_list = []
    for k in range(temp_every_len):

        m = temp_every_len - k   #反向循环，从最高位到最低位开始遍历切分  3

        beishu = 1

        for u in range(m-1):

            beishu = beishu * 10

            #print(u,m,beishu)
        #print(beishu)

        weishu = int(now_num/beishu)
        #print(weishu)
        temp_list.append(weishu)
        #print(temp_list)
        now_num = now_num - (weishu*beishu)

    last_list.append(temp_list)#拿到最终列表  7 13 4 246 的最终列表是  [[7],[1,3]...[2,4,6]]


#print(last_list_len)
last_list_len.sort(reverse=True)
maxlen = last_list_len[0]
#print(last_list_len)
temp_list_sort = last_list
#print("num",num)
res=num

for i in range(N):   #位数补全 全部为最高位数

    if len(temp_list_sort[i]) == int(maxlen):
        #print("aa",len(temp_list_sort[i]),maxlen)
        continue
    else:
        #print("bb", len(temp_list_sort[i]), maxlen)
        #print("i",i)
        for k in range(maxlen-len(temp_list_sort[i])):
            num1[i] *=10

# print("num1",num1)
# print("res",res)

b=sorted(enumerate(num1), key=lambda x:x[1],reverse=True)
#print(b)
laststr = ""

for k in range(N):

    laststr = laststr + str(num[b[k][0]])

    #print("11",laststr)
print(laststr)