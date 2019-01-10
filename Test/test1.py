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

# N = int(input())
N = 2

# num = input()
num = '12 456'
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



# print(last_list_len)
a=copy.deepcopy(last_list_len)
a.sort(reverse=True)
maxlen = a[0]
#print(last_list_len)
temp_list_sort = last_list
#print("num",num)
res=num
a= 1
for i in range(maxlen):

    a=a*10

maxlen1 = a/10

print(maxlen1)

# for i in range(N):   #位数补全 全部为最高位数
#     every_len = last_list_len[i]
#     if int(every_len/maxlen1) >= 1:
#         #print("aa",len(temp_list_sort[i]),maxlen)
#         continue
#     else:
#         #print("bb", len(temp_list_sort[i]), maxlen)
#         #print("i",i)
#         for k in range(int(maxlen1-every_len)):
#             print(int(maxlen1-every_len))
#             num1[i] *=10

# print("num1",num1)
# print("res",res)

b=sorted(enumerate(num1), key=lambda x:x[1],reverse=True)
#print(b)
laststr = ""

for k in range(N):

    laststr = laststr + str(num[b[k][0]])

    #print("11",laststr)
print(laststr)