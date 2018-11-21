#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : master.py
# @Author: WangYe
# @Date  : 2018/11/21
# @Software: PyCharm

'''
处理分配worker的容器
'''
class master_1(object):
    def __init__(self,batch_size,num_worker,mini_num,num_data):
        #mini_num是指将大的batch_size切成多少小的样本
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.minu_num = mini_num
        self.num_data = num_data
        #self.*woker = *woker
    def input_check(self,*worker_id):
        #计算完成检测
        #list_worker = []
        list_worker = worker_id
        return list_worker

    def have_computed(self):
        #已经计算过的样本总和
        self.num_data = self.num_data +1
        return self.num_data
    def rest_check(self):
        #剩余样本检测
        mini_num = self.minu_num
        num_now = master_1.have_computed(self)
        rest = mini_num - num_now
        '''
        如果还有剩余样本，返回1，否则0
        '''
        if rest > 0:
            return 1
        else:
            return 0

    def return_check(self,ip):
        #检测是否将下一次样本分配给worker(ip)
        temp_rest = master_1.rest_check(self)
        if  temp_rest == 1:
            #如果还有样本
            have_finished_woker=master_1.input_check(self)  #输入IP是否在已完成worker列表当中
            if ip in have_finished_woker:
                return 1
            else:
                return 0
        else:
            return -1  #本次批次训练完成，可以传送梯度给ps

