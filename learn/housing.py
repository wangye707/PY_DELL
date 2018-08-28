#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : housing.py
# @Author: WangYe
# @Date  : 2018/4/17
# @Software: PyCharm
# 房价估算（决策树预测回归模型）
import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor #决策树
from sklearn.ensemble import AdaBoostRegressor  #AdaBoostRegressor用于回归，AdaBoostClassifier用于分类
from sklearn import datasets #数据下载？
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
housing_data=datasets.load_boston()         #scikit_learn提供的数据接口，下载数据
x,y=shuffle(housing_data.data,housing_data.target,random_state=7) #shuffle打乱数据顺序
#数据级分类，80%的数据集，20%的测试数据
num_training=int(0.8*len(x))
x_train,y_train=x[:num_training],y[:num_training]
x_test,y_test=x[num_training:],y[num_training:]
#拟合决策树回归模型，选一个最大深度为4的决策树，限制决策树不变成任意深度
#建立决策树模型 最大深度是4 限制决策树的深度
dt_regressor=DecisionTreeRegressor(max_depth=4)  #树深度为4
dt_regressor.fit(x_train,y_train)  #拟合x和y的测试集
#使用带AdaBoost算法的决策树模型进行拟合 fit代表拟合
#帮助我们比对效果，看看AdaBoost算法对决策树回归器的训练效果有多大改善
ab_regressor=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400,random_state=7)
ab_regressor.fit(x_train,y_train)
#查看决策树回归器的训练结果
y_pred_dt=dt_regressor.predict(x_test)
mse=mean_squared_error(y_test,y_pred_dt)
evs=explained_variance_score(y_test,y_pred_dt)
print("\n ###决策树学习效果###")
print("均方误差 = ",round(mse,2))
print("解释方差分 = ",round(evs,2))
#查看对AdaBoost进行进行改进之后的算法
y_pred_ab=ab_regressor.predict(x_test)
mse=mean_squared_error(y_test,y_pred_ab)
evs=explained_variance_score(y_test,y_pred_ab)
print("\n ###AdaBoost算法改善效果###")
print("均方误差 = ",round(mse,2))
print("解释方差分 = ",round(evs,2))
def plot_feature_importances(feature_importances,title,feature_names):
    #将重要性值标准化
    feature_importances=100.0*(feature_importances/max(feature_importances))
    #将得分从低到高进行排序 flipud 实现矩阵的翻转
    index_sorted=np.flipud(np.argsort(feature_importances))
    #让x坐标轴上的把标签居中 shape用于读取矩阵的长度
    pos=np.arange(index_sorted.shape[0])+0.5
    #画出条形图
    plt.figure()
    #bar代表柱形图
    plt.bar(pos,feature_importances[index_sorted],align='center')
    #为x轴的主刻度设置值
    plt.xticks(pos,feature_names[index_sorted])
    plt.ylabel('Relative importance')
    plt.title(title)
    plt.show()
#特征重要性 图形表示 feature_importances 代表每个特征多样性
#plot_feature_importances(dt_regressor.feature_importances_,'Decision Tree Regressor ',housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_,'AdaBoost Regressor ',housing_data.feature_names)