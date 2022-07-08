#!/usr/bin/env python
# coding: utf-8

# In[58]:


#1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv("SteelPlateFaults-2class.csv")
df0=df[df['Class']==0]
df1=df[df['Class']==1]
X=df.columns
#creating columns for training data
Y=[]
for i in X:
    if not(i=='Class'):
        Y.append(i)
X_train0, X_test0, X_label_train0, X_label_test0= train_test_split(df0[Y] ,df0['Class'], test_size=0.3, random_state=42, shuffle=True)
X_train1, X_test1, X_label_train1, X_label_test1= train_test_split(df1[Y] ,df1['Class'], test_size=0.3, random_state=42, shuffle=True)
#saving train and test data to csv file
X_train=pd.concat([X_train0,X_train1])
X_test=pd.concat([X_test0,X_test1])
X_label_train=pd.concat([X_label_train0,X_label_train1])
X_label_test=pd.concat([X_label_test0,X_label_test1])
X_train.to_csv("SteelPlateFaults-train.csv", index = False)
X_test.to_csv("SteelPlateFaults-test.csv", index = False)
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix as cnf_mat
from sklearn.metrics import accuracy_score as accuracy
l=[1,3,5]
# finding cnfusion matrix and accuracy
for i in l:
    KNN_class=KNN(n_neighbors=i)
    KNN_class.fit(X_train, X_label_train)
    pred=KNN_class.predict(X_test)
    print("\nconfusion matrix for k=",i,"::\n")
    print(cnf_mat(X_label_test,pred))
    print("accuracy for k=",i,"::")
    print(round(accuracy(pred,X_label_test)*100,2),"%")


# In[59]:


#2
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - X_train[column].min()) / (X_train[column].max() - X_train[column].min())
    return df_norm
X_train=pd.read_csv("SteelPlateFaults-train.csv")
X_test=pd.read_csv("SteelPlateFaults-test.csv")
#normalising the data with min max normalisation
X_train1=min_max_scaling(X_train)
X_test1=min_max_scaling(X_test)
#finding accuracy and confusion matrix
for i in l:
    KNN_class=KNN(n_neighbors=i)
    KNN_class.fit(X_train1, X_label_train)
    pred=KNN_class.predict(X_test1)
    print("\nconfusion matrix for k=",i,"::\n")
    print(cnf_mat(X_label_test,pred))
    print("accuracy for k=",i,"::")
    print(round(accuracy(pred,X_label_test)*100,2),"%")


# In[65]:


#3
#dropping the columns with high correlation
df_mod=df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400','Class'],axis=1)
df_s=min_max_scaling(df_mod)
Y=[]
column=df['Class']
X=df_s.columns
for i in X:
    if not(i=='Class'):
        Y.append(i)
X_train, X_test, X_label_train, X_label_test= train_test_split(df_s[Y] ,column, test_size=0.3, random_state=42, shuffle=True)
X_train0 = X_train[X_label_train== 0]
X_train1 = X_train[X_label_train== 1] 
#covariance and mean claculation
cov_0 = np.cov(X_train0.T)
cov_1 = np.cov(X_train1.T)                                   
mean_0 = np.mean(X_train0)
mean_1 = np.mean(X_train1)
#prior probability calculation
prior_prob0 = len(X_train0)/763
prior_prob1 = len(X_train1)/763
#likelihood calculation
def likeli_calc(data, mean, cov):                                 
    return np.exp(-0.5*np.dot(np.dot((data-mean).T, np.linalg.inv(cov)), (data-mean)))/((2*np.pi)*5 * (np.linalg.det(cov))*0.5)
predicted_class = []
for i in range(len(X_test)):                              
    prob_for_0=likeli_calc(X_test.iloc[i],mean_0,cov_0)*prior_prob0
    prob_for_1=likeli_calc(X_test.iloc[i],mean_1,cov_1)*prior_prob1
    if prob_for_0>=prob_for_1:
        predicted_class.append(0)
    else:
        predicted_class.append(1)
#printing confusion matrix and accuracy
print("confusion matrix for the above classifier is::\n",cnf_mat(X_label_test,predicted_class))
print("% age accuracy::\n",round(accuracy(predicted_class,X_label_test)*100,2),"%")


# In[ ]:




