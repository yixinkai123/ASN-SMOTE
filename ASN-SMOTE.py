# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:25:10 2021

@author: 易新凯
"""
#NM-Smote

#分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
#标准化工具
from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#导入集合分割，交叉验证，网格搜索
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,cross_validate,KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve,auc,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#smote过采样
# from imblearn.over_sampling import SMOTE,BorderlineSMOTE,SVMSMOTE,ADASYN,KMeansSMOTE,RandomOverSampler
# #欠采样
# from imblearn.under_sampling import RandomUnderSampler
import random
import math
import heapq
import time
def Euclidean_Metric(a,b):
      """
      欧式距离
      """
      dis=0
      A = np.array(a)
      B = np.array(b)
      n=A.shape[0]
      for i in range(n):
          dis=dis+(A[i]-B[i])*(A[i]-B[i])
      dis=np.sqrt(dis)
      return dis
def generate_x(samples,N,k):
    #n=int(N/10)
    time_start=time.time()
    g_index=0
    wrg=0
    samples_X=samples.iloc[:,0:-1]
    samples_Y=samples.iloc[:,-1]
    Minority_sample=samples[samples.iloc[:,-1].isin(['1'])]
    Minority_sample_X=Minority_sample.iloc[:,0:-1]
                                       
    # transfer = StandardScaler()
    # SMinority_X= transfer.fit_transform(Minority_sample)
    # All_X=transfer.fit_transform(samples_X)
    Minority_X=np.array(Minority_sample_X)
    All_X=np.array(samples_X)
    n1=All_X.shape[0]-2*Minority_X.shape[0]
    print(n1)
    #n=int((All_X.shape[0]-2*Minority_X.shape[0])/Minority_X.shape[0])
    #print(n)
    dis_matrix=np.zeros((Minority_X.shape[0],All_X.shape[0]),dtype=float)
    for i in range(0,Minority_X.shape[0]):
        for j in range(0,All_X.shape[0]):
            dis_matrix[i,j]=Euclidean_Metric(Minority_X[i,:],All_X[j,:])
            if(dis_matrix[i,j]==0):
                dis_matrix[i,j]=999999
    dis_matrix=dis_matrix.tolist()
    
    d=[]
    #print(Minority_X.shape[0])
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(1, dis_matrix[i])))
        #print(min_index)
        if(samples_Y[min_index[0]]==0): 
            d.append(i)
    Minority_X=np.delete(Minority_X,d,axis=0)
    #print(Minority_X.shape)
    n=int((n1)/Minority_X.shape[0])
    #print(n)
    synthetic = np.zeros(((Minority_X.shape[0])*n,Minority_X.shape[1]),dtype=float)
    #print(Minority_X.shape[0])
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(k, dis_matrix[i])))
        best_index={}
        best_f=0
        for h in range(len(min_index)):
            
            if(samples_Y[min_index[h]]==0):
               best_index[best_f]=min_index[h]
               best_f+=1
               break
            else:
                best_index[best_f]=min_index[h]
                best_f+=1
        #print(best_index)
        for j in range(0,n):
            nn=random.randint(0,len(best_index)-1)
            #print(min_index[nn])
            dif=All_X[best_index[nn]]-Minority_X[i]
            #print(dif)
            gap=random.random()
            synthetic[g_index]=Minority_X[i]+gap*dif
            g_index+=1
            
    #print(synthetic.shape)
    #print(wrg)
    
    # synthetic=synthetic[0:synthetic.shape[0]-,:]
    labels=np.ones(synthetic.shape[0])
    synthetic=np.insert(synthetic,synthetic.shape[1],values=labels,axis=1)
    examples=np.concatenate((samples,synthetic),axis=0)
    time_end=time.time()
    del(dis_matrix)
    return examples
def knn_classifier(xtrain,ytrain,xtest,ytest):
    knn=KNeighborsClassifier(n_neighbors=3 ,weights='distance',p=1 )
    # params={'n_neighbors':[i for i in range(1,10)],
    #         'weights':['uniform','distance'],
    #       'p':[1,2]}
    #gcv=GridSearchCV(knn,params,scoring='roc_auc',cv=10)
    knn=knn.fit(xtrain,ytrain)
    # 得到最好的模型，进行预测
    #knn_best=gcv.best_estimator_
    # y_=knn_best.predict(xtest)
    
    cm=confusion_matrix(ytest,knn.predict(xtest))
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    AUC=roc_auc_score(ytest,knn.predict_proba(xtest)[:,1])
    #Acc=(TP+TN)/(TP+TN+FP+FN)
    Pos_Precision=TP/(TP+FP)
    #Neg_Precision=TN/(TN+FN)
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    F_Measure=(2*Sensitivity*Pos_Precision)/(Sensitivity+Pos_Precision)
    G_Mean=np.sqrt(Sensitivity*Specificity)
    # print("F_Measure=%.6f" % F_Measure)
    # print("G_Mean=%.6f" %G_Mean)
    # print("AUC=%.6f" %AUC)
    # print("Acc=%.6f" % Acc
    return F_Measure,G_Mean,AUC
def LRClassifier(xtrain,ytrain,xtest,ytest):
    log = LogisticRegression(penalty='l2',C=1.0,max_iter=1000)
    # params={'penalty':['l1','l2'],
    #         'C':[1,2,3,4,5],
    #         }
    #gcv=GridSearchCV(log,params,scoring='roc_auc',cv=10)
    log=log.fit(xtrain,ytrain)
    #log_best=gcv.best_estimator_
    
    cm=confusion_matrix(ytest,log.predict(xtest))
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    AUC=roc_auc_score(ytest,log.predict_proba(xtest)[:,1])
    Pos_Precision=TP/(TP+FP)
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    F_Measure=2*Sensitivity*Pos_Precision/(Sensitivity+Pos_Precision)
    G_Mean=np.sqrt(Sensitivity*Specificity)
    # print("F_Measure=%.6f" % F_Measure)
    # print("G_Mean=%.6f" %G_Mean)
    # print("AUC=%.6f" %AUC)
    print("Acc=%.6f" % Acc)
    return F_Measure,G_Mean,AUC
def RandomforClassifier(xtrain,ytrain,xtest,ytest):
    transfer = StandardScaler()
    xtrain = transfer.fit_transform(xtrain)
    xtest = transfer.transform(xtest)
    #选用随机森林模型
    rfc=RandomForestClassifier(
                                criterion='gini',
                                n_estimators=100,
                                min_samples_split=2,
                                min_samples_leaf=2,
                                max_depth=15,
                                random_state=6)
    #score_pre = cross_val_score(rfc,xtrain,ytrain,scoring='roc_auc',cv=10).mean()
    #scores = cross_val_score(rfc,xtrain,ytrain,cv=10,scoring='roc_auc')
    #print(scores)
    #print('mean CV-Scores: %.6f' % score_pre)
    rfc=rfc.fit(xtrain,ytrain)
    # #测试评估
    #result=rfc.score(xtest,ytest)
    AUC=roc_auc_score(ytest,rfc.predict_proba(xtest)[:,1])
    cm=confusion_matrix(ytest,rfc.predict(xtest))
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    Acc=(TP+TN)/(TP+TN+FP+FN)
    Pos_Precision=TP/(TP+FP)
    #print("%.3f" %(Pos_Precision))
    #Neg_Precision=TN/(TN+FN)
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    F_Measure=2*Sensitivity*Pos_Precision/(Sensitivity+Pos_Precision)
    G_Mean=np.sqrt(Sensitivity*Specificity)
    #print("F_Measure=%.6f" % F_Measure)
    #print("G_Mean=%.6f" %G_Mean)
    #print("AUC=%.6f" %AUC)
    #print("Acc=%.6f" % Acc)
    return F_Measure,G_Mean,AUC,Acc
def SvmClassifier(xtrain,ytrain,xtest,ytest):
    SVM = svm.SVC(probability=True,degree=2)
    # params={'kernel':['poly','rbf'],
    #         'C':[0.001,0.01,0.1,1],C=0.1,kernel='',
    #         'gamma':[0.1,0.3,0.5,0.7,1,2,5]}
    #gcv=GridSearchCV(SVM,params,scoring='roc_auc',cv=10)
    SVM=SVM.fit(xtrain,ytrain)
    #svm_best=gcv.best_estimator_
    # predictions_validation = svm.predict_proba(xtest)[:,1]
    # fpr, tpr, _=roc_curve(ytest, predictions_validation)
    # roc_auc = auc(fpr, tpr)
    # score_pre = cross_val_score(svm,xtrain,ytrain,scoring='roc_auc',cv=10).mean()
    # scores = cross_val_score(svm,xtrain,ytrain,cv=10,scoring='roc_auc')
    cm=confusion_matrix(ytest,SVM.predict(xtest))
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    AUC=roc_auc_score(ytest,SVM.predict(xtest))
    #Acc=(TP+TN)/(TP+TN+FP+FN)
    Pos_Precision=TP/(TP+FP)
    #Neg_Precision=TN/(TN+FN)
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    F_Measure=2*Sensitivity*Pos_Precision/(Sensitivity+Pos_Precision)
    G_Mean=np.sqrt(Sensitivity*Specificity)
    #print("F_Measure=%.6f" % F_Measure)
    # print("G_Mean=%.6f" %G_Mean)
    # print("AUC=%.6f" %AUC)
    # print("Acc=%.6f" % Acc)
    return F_Measure,G_Mean,AUC
def LDAClassifier(xtrain,ytrain,xtest,ytest):
    clf = LinearDiscriminantAnalysis()
    clf=clf.fit(xtrain, ytrain)
    cm=confusion_matrix(ytest,clf.predict(xtest))
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    AUC=roc_auc_score(ytest,clf.predict(xtest))
    #Acc=(TP+TN)/(TP+TN+FP+FN)
    Pos_Precision=TP/(TP+FP)
    #Neg_Precision=TN/(TN+FN)
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    F_Measure=2*Sensitivity*Pos_Precision/(Sensitivity+Pos_Precision)
    G_Mean=np.sqrt(Sensitivity*Specificity)
    #print("F_Measure=%.6f" % F_Measure)
    # print("G_Mean=%.6f" %G_Mean)
    # print("AUC=%.6f" %AUC)
    # print("Acc=%.6f" % Acc)
    return F_Measure,G_Mean,AUC
def NBClassifier(xtrain,ytrain,xtest,ytest):
    gnb=GaussianNB()
    gnb=gnb.fit(xtrain, ytrain)
    cm=confusion_matrix(ytest,gnb.predict(xtest))
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    AUC=roc_auc_score(ytest,gnb.predict(xtest))
    #Acc=(TP+TN)/(TP+TN+FP+FN)
    Pos_Precision=TP/(TP+FP)
    #Neg_Precision=TN/(TN+FN)
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    F_Measure=2*Sensitivity*Pos_Precision/(Sensitivity+Pos_Precision)
    G_Mean=np.sqrt(Sensitivity*Specificity)
    #print("F_Measure=%.6f" % F_Measure)
    # print("G_Mean=%.6f" %G_Mean)
    # print("AUC=%.6f" %AUC)
    # print("Acc=%.6f" % Acc)
    return F_Measure,G_Mean,AUC
plt.rcParams['font.sans-serif']=['Times New Roman']
data = pd.read_excel('DFS-B超&MRI.xlsx',encoding='gbk')

#g_sample=generate_x(data,100,5)

print("ASN_SMOTE")
X = data.drop(['DFS'],axis=1)
Y = data['DFS']
x=np.array(X)
y=np.array(Y)
kf=StratifiedKFold(n_splits=5)
F=[]
G=[]
A=[]
T=[]
Ac=[]
for train_index,test_index in kf.split(x,y):
    
    #print((train_index,test_index))
    #print("~~")
    xtrain,xtest=x[train_index],x[test_index]
    
    ytrain,ytest=y[train_index],y[test_index]
    kdata=pd.DataFrame(np.column_stack((xtrain,ytrain)))
    # #print(kdata.shape)
    time_start=time.time() 
    g_sample=generate_x(kdata,100,7)
    time_end=time.time()
    s_time=time_end-time_start
    # x=g_sample[:,0:-1]
    # y=g_sample[:,-1]
    afm=[]
    agm=[]
    aAuc=[]
    aAcc=[]
    for i in range(3):
        fm,gm,Auc,Acc=RandomforClassifier(g_sample[:,0:-1],g_sample[:,-1],xtest,ytest)
        afm.append(fm)
        agm.append(gm)
        aAuc.append(Auc)
        aAcc.append(Acc)
    F.append(np.mean(afm))
    G.append(np.mean(agm))
    A.append(np.mean(aAuc))
    Ac.append(np.mean(aAcc))
print("%.3f±%.3f" %(np.mean(G),np.std(G)))
print("%.3f±%.3f" %(np.mean(F),np.std(F)))
print("ASN-SMOTE")
print("DFS-B超&MRIwA")
print("AUC=%.3f±%.3f" %(np.mean(A),np.std(A)))
print("ACC=%.3f±%.3f" %(np.mean(Ac),np.std(Ac)))
#print("%.4f ± %.4f" %(np.mean(T),np.std(T)))
