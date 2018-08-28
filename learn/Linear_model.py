#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : Linear_model.py
# @Author: WangYe
# @Date  : 2018/6/20
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,discriminant_analysis,cross_validation,model_selection
def load_data():
    #载入糖尿病病人数据集。数据集有442个样本
    #每个样本是个特征
    #每个特征是浮点数    -0.2到0.2之间
    #样本的目标都在25到346之间
    diabetes=datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,diabetes.target,
                                             test_size=0.25,random_state=0)
def test_LinearRegression(*data):  #线性回归模型
    X_train,X_test,y_train,y_test=data
    regr=linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    print("LinearRegression回归")
    print('coefficient:%s,intercept%.2f'%(regr.coef_,regr.intercept_))  #coef_是权重向量，intercept是b纸
    print("Residual sum of squares: %.2f"% np.mean((regr.predict(X_test)-y_test)**2))#模型预测返回，mean为均值
    print('score: %.2f' %regr.score(X_test,y_test))   #性能得分
X_train,X_test,y_train,y_test=load_data()
test_LinearRegression(X_train,X_test,y_train,y_test)
#socre的值不会超过1，但可能为负数。值越大意味着性能越好
#第一步预测到此结束
def test_Ridge(*data):  #岭回归（使模型更加稳健）
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    print("Ridge回归")
    print('coefficient:%s,intercept%.2f' % (regr.coef_, regr.intercept_))  # coef_是权重向量，intercept是b纸
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))  # 模型预测返回，mean为均值
    print('score: %.2f' % regr.score(X_test, y_test))  # 性能得分
X_train, X_test, y_train, y_test = load_data()
test_Ridge(X_train, X_test, y_train, y_test)
#针对不同的α值对于预测性能的影响测试
#第二步结束
def test_Ridge_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores=[]
    for i,alpha in enumerate(alphas):
        regr=linear_model.Ridge(alpha=alpha)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_test,y_test))
    #绘图
    fig=plt.figure()   #作用新建绘画窗口,独立显示绘画的图片
    ax=fig.add_subplot(1,1,1)
    # 这个比较重要,需要重点掌握,参数有r,c,n三个参数
    # 使用这个函数的重点是将多个图像画在同一个绘画窗口.
    # r  表示行数
    # c  表示列行
    # n  表示第几个
    ax.plot(alphas,scores)   #添加参数
    ax.set_xlabel(r"$\alpha$")  #转换α符号
    ax.set_ylabel(r"score")
    ax.set_xscale('log')  #x轴转换为对数
    ax.set_title("Ridge")
    plt.show()
X_train, X_test, y_train, y_test = load_data()
test_Ridge_alpha(X_train, X_test, y_train, y_test)
#第三步结束
#Lasso回归
#lasso回归可以将系数控制收缩到0，从而达到变量选择的效果，这是一种非常流行的选择方法
def test_Lasso(*data):  #岭回归（使模型更加稳健）
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Lasso()
    regr.fit(X_train, y_train)
    print("Lasso回归")
    print('coefficient:%s,intercept%.2f' % (regr.coef_, regr.intercept_))  # coef_是权重向量，intercept是b纸
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))  # 模型预测返回，mean为均值
    print('score: %.2f' % regr.score(X_test, y_test))  # 性能得分
X_train, X_test, y_train, y_test = load_data()
test_Lasso(X_train, X_test, y_train, y_test)
#岭回归结束结束
#针对岭回归，我们也采用不同的α值进行测试
def test_Lasso_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    scores=[]
    for i,alpha in enumerate(alphas):
        regr=linear_model.Lasso(alpha=alpha)
        regr.fit(X_train,y_train)
        scores.append(regr.score(X_test,y_test))
    #绘图
    fig=plt.figure()   #作用新建绘画窗口,独立显示绘画的图片
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,scores)   #添加参数
    ax.set_xlabel(r"$\alpha$")  #转换α符号
    ax.set_ylabel(r"score")
    ax.set_xscale('log')  #x轴转换为对数
    ax.set_title("Lasso")
    plt.show()
X_train, X_test, y_train, y_test = load_data()
test_Lasso_alpha(X_train, X_test, y_train, y_test)
#岭回归α值测试结束
#ElasticNet回归测试
def test_ElasticNet(*data):  #岭回归（使模型更加稳健）
    X_train, X_test, y_train, y_test = data
    regr = linear_model.ElasticNet()
    regr.fit(X_train, y_train)
    print("ElasticNet回归")
    print('coefficient:%s,intercept%.2f' % (regr.coef_, regr.intercept_))  # coef_是权重向量，intercept是b纸
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))  # 模型预测返回，mean为均值
    print('score: %.2f' % regr.score(X_test, y_test))  # 性能得分
X_train, X_test, y_train, y_test = load_data()
test_ElasticNet(X_train, X_test, y_train, y_test)
#ElasticNet预测回归结束
#针对ElasticNet不同的α,ρ值的测试
def test_ElasticNet_alpha_rho(*data):
    X_train, X_test, y_train, y_test = data
    alphas=np.logspace(-2,2)    #α值区间
    rhos=np.linspace(0.01,1)    #ρ值区间
    scores=[]
    for alpha in alphas:
        for rho in rhos:
            regr=linear_model.ElasticNet(alpha=alpha,l1_ratio=rho)   #针对α和ρ的值来判断最终值
            regr.fit(X_train,y_train)
            scores.append(regr.score(X_test,y_test))
    #绘图
    alphas,rhos=np.meshgrid(alphas,rhos)
    scores=np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig=plt.figure()   #作用新建绘画窗口,独立显示绘画的图片
    ax=Axes3D(fig)
    surf=ax.plot_surface(alphas,rhos,scores,rstride=1,cstride=1,
                         cmap=cm.jet,linewidth=0,antialiased=False)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    ax.set_xlabel(r"$\alpha$")  #转换α符号
    ax.set_ylabel(r"$\rho$")  # 转换ρ符号
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()
X_train, X_test, y_train, y_test = load_data()
test_ElasticNet_alpha_rho(X_train, X_test, y_train, y_test)
#NlasticNet针对α和ρ的变化的三维坐标完成
#逻辑回归，首先先将载入函数进行修改
def load_data1():
    iris=datasets.load_iris()
    X_train=iris.data
    y_train=iris.target
    return cross_validation.train_test_split(X_train,y_train,test_size=0.25,random_state=0,stratify=y_train)
#数据集前50个样本类别为0，中间50个位1，后50个位2
def test_LogisticRegress(*data):
    X_train,X_test,y_train,y_test=data
    regr=linear_model.LogisticRegression()
    regr.fit(X_train,y_train)
    print("LogisticRegress逻辑回归")
    print('coefficient:%s,intercept%s' % (regr.coef_, regr.intercept_))  # coef_是权重向量，intercept是b纸
   # print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))  # 模型预测返回，mean为均值
    print('score: %.2f' % regr.score(X_test, y_test))  # 性能得分
X_train, X_test, y_train, y_test = load_data1()
test_LogisticRegress(X_train, X_test, y_train, y_test)

