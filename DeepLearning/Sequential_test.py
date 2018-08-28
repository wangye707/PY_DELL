import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn import feature_selection
import skflow
import tensorflow as tf
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
import keras

#matlab文件名  
matfn=u'E:/LQPHP/audiotrack/ASD实验数据/lqp_14种动态手势数据/FisherScore05.mat'
data=sio.loadmat(matfn)  
 


train = data['ttz']
target = data['label']



train_X,test_X, train_y, test_y = train_test_split(train,  
                                                   target,
                                                   test_size = 0.5,
                                                   random_state = 0)
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(train_X)
X_test_norm = mms.fit_transform(test_X)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(train_X)
X_test_std = stdsc.transform(test_X)

X_train_std = X_train_std
X_test_std = X_test_std

#将一维label转换为二维label
train_Y=keras.utils.to_categorical((train_y-1), num_classes=14)
test_Y = keras.utils.to_categorical((test_y-1), num_classes=14)

#https://keras.io/zh/optimizers/   中文官方文档
#input_dim  为特征维度个数
model = Sequential()
model.add(Dense(12,input_dim=X_train_std.shape[1],init='uniform',activation='relu'))
#model.add(Dense(80,input_dim=X_train_std.shape[1],init='uniform',activation='softmax'))
#model.add(Dense(5,input_dim=X_train_std.shape[1],init='uniform',activation='sigmoid'))
model.add(Dense(14,input_dim=X_train_std.shape[1],init='uniform',activation='sigmoid'))
#损失函数：loss='mean_squared_error','mean_absolute_error','mean_absolute_percentage_error','sparse_categorical_crossentropy','binary_crossentropy'
#   loss = 'mean_squared_logarithmic_error','squared_hinge','hinge','categorical_hinge','logcosh','categorical_crossentropy','kullback_leibler_divergence'
# loss='poisson','cosine_proximity'
#optimizer='adam','SGD','RMSprop','Adagrad','Adadelta','Adamax','Nadam','TFOptimizer'
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train_std,train_Y,nb_epoch=500,batch_size=150)
scoress=model.evaluate(X_test_std,test_Y)
scoress_train = model.evaluate(X_train_std,train_Y)


print('Score:',scoress[1])
print('Score_train:',scoress_train)
