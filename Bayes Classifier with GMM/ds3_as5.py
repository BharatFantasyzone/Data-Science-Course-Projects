#!/usr/bin/env python
# coding: utf-8

# In[2]:


#part B
#Q1
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as accuracy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
df=pd.read_csv("abalone.csv")
col=[]
for i in df.columns:
    if(i!="Rings"):
        col.append(i)
X_train, X_test, Y_train, Y_test = train_test_split( df[col], df["Rings"], test_size=0.3, random_state=42)
X_train1=pd.concat([X_train,Y_train])
X_test1=pd.concat([X_test,Y_test])
X_train.to_csv("abalone_train.csv", index = False)
X_test.to_csv("abalone_test.csv", index = False)
dict_={}
for i in X_train.columns:
        dict_[i]=np.corrcoef(df[i],df["Rings"])[0][1]
req_atr=max(zip(dict_.values(), dict_.keys()))[1]
reg = LinearRegression().fit(X_train[[req_atr]], Y_train)
#on testing data
Y_pred_test= reg.predict(X_test[[req_atr]]) 
print("RMSE on testing data",np.sqrt(mse(Y_pred_test,Y_test)))
#on training data
Y_pred_train= reg.predict(X_train[[req_atr]]) 
print("RMSE on training data",np.sqrt(mse(Y_pred_train,Y_train)))


# In[3]:


import matplotlib.pyplot as plt
plt.scatter(X_train[[req_atr]],Y_train)
plt.plot(X_test[[req_atr]],Y_pred_test,color="g")
plt.xlabel(req_atr)
plt.ylabel("Rings")
plt.show()
plt.scatter(Y_test,Y_pred_test)
plt.xlabel("actual rings")
plt.ylabel("predicted rings")


# In[4]:


#Q2
reg = LinearRegression().fit(X_train[col], Y_train)
#on testing data
Y_pred_test= reg.predict(X_test[col]) 
print("RMSE on testing data",np.sqrt(mse(Y_pred_test,Y_test)))
#on training data
Y_pred_train= reg.predict(X_train[col]) 
print("RMSE on training data",np.sqrt(mse(Y_pred_train,Y_train)))


# In[5]:


plt.scatter(Y_test,Y_pred_test)
plt.xlabel("actual rings")
plt.ylabel("predicted rings")


# In[56]:


#3
df=pd.read_csv("abalone.csv")
col=[]
for i in df.columns:
    if(i!="Rings"):
        col.append(i)
X_train, X_test, Y_train, Y_test = train_test_split( df[col], df["Rings"], test_size=0.3, random_state=42)
rmse_train=[]
rmse_test=[]
l=[2,3,4,5]
for i in l:
    poly_features = PolynomialFeatures(i) #p is the degree
    x_poly = poly_features.fit_transform(X_train[[req_atr]])
    regressor = LinearRegression()
    regressor.fit(x_poly, Y_train) 
    #Input arguments: x_poly: Polynomial expansion of input 
    Y_pred_train = regressor.predict(poly_features.fit_transform(X_train[[req_atr]]))
    rmse_train.append(np.sqrt(mse(Y_pred_train,Y_train)))
    Y_pred_test = regressor.predict(poly_features.fit_transform(X_test[[req_atr]]))
    if(i==4):
        Y_pred_test4=Y_pred_test
    rmse_test.append(np.sqrt(mse(Y_pred_test,Y_test)))
plt.bar(l,rmse_train)
plt.title("for train")
plt.ylabel("RMSE")
plt.xlabel("degree")
plt.show()
plt.bar(l,rmse_test)
plt.title("for test")
plt.ylabel("RMSE")
plt.xlabel("degree")
plt.show()
plt.scatter(X_train[[req_atr]],Y_train)
plt.scatter(X_test[[req_atr]],Y_pred_test4,color="g")
plt.xlabel(req_atr)
plt.ylabel("Rings")
plt.title("best fit curve")
plt.show()
plt.scatter(Y_test,Y_pred_test4)
plt.title("prediction from best fit curve")
plt.xlabel("actual rings")
plt.ylabel("predicted rings")


# In[60]:


#4
df=pd.read_csv("abalone.csv")
col=[]
for i in df.columns:
    if(i!="Rings"):
        col.append(i)
X_train, X_test, Y_train, Y_test = train_test_split( df[col], df["Rings"], test_size=0.3, random_state=42)
rmse_train=[]
rmse_test=[]
l=[2,3,4,5]
for i in l:
    poly_features = PolynomialFeatures(i) #p is the degree
    x_poly = poly_features.fit_transform(X_train[col])
    regressor = LinearRegression()
    regressor.fit(x_poly, Y_train) 
    #Input arguments: x_poly: Polynomial expansion of input 
    Y_pred_train = regressor.predict(poly_features.fit_transform(X_train[col]))
    rmse_train.append(np.sqrt(mse(Y_pred_train,Y_train)))
    Y_pred_test = regressor.predict(poly_features.fit_transform(X_test[col]))
    if(i==2):
        Y_pred_test4=Y_pred_test
    rmse_test.append(np.sqrt(mse(Y_pred_test,Y_test)))
plt.bar(l,rmse_train)
plt.title("for train")
plt.ylabel("RMSE")
plt.xlabel("degree")
plt.show()
plt.bar(l,rmse_test)
plt.title("for test")
plt.ylabel("RMSE")
plt.xlabel("degree")
plt.show()
plt.scatter(Y_test,Y_pred_test4)
plt.title("prediction from best fit curve")
plt.xlabel("actual rings")
plt.ylabel("predicted rings")


# In[81]:

#part A
#Q1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Part A
#Question 1
df=pd.read_csv("SteelPlateFaults-2class.csv")
df=df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400','Class'],axis=1)
X_train0, X_test0, X_label_train0, X_label_test0= train_test_split(df0[Y] ,df0['Class'], test_size=0.3, random_state=42, shuffle=True)
X_train1, X_test1, X_label_train1, X_label_test1= train_test_split(df1[Y] ,df1['Class'], test_size=0.3, random_state=42, shuffle=True)

x_train = pd.concat([X_train0, X_train1], axis=0)
y_train = pd.concat([X_label_train0, X_label_train1],axis=0)

xtrain0 = X_train0
xtrain1 = X_train1
x_test = pd.concat([X_test0, X_test1], axis=0)
y_test = pd.concat([X_label_test0, X_label_test1],axis=0)

for q in [2, 4, 8, 16]:
    if(q==2 or q==4):
        gmm0 = GaussianMixture(n_components = q, covariance_type = "full", random_state = 42)
        gmm1 = GaussianMixture(n_components = q, covariance_type = "full", random_state = 42)
    else:
        gmm0 = GaussianMixture(n_components = q, covariance_type = "full", random_state = 42,reg_covar=np.exp(-5))
        gmm1 = GaussianMixture(n_components = q, covariance_type = "full", random_state = 42,reg_covar=np.exp(-5))
    
    gmm0.fit(xtrain0)
    gmm1.fit(xtrain1)
    likelihood0 = gmm0.score_samples(x_test) + np.log(len(train0)/len(train))
    likelihood1 = gmm1.score_samples(x_test) + np.log(len(train1)/len(train))
    predict = []
    for i in range(len(x_test)):
        if likelihood0[i] > likelihood1[i]:
            predict.append(0)
        else:
            predict.append(1)
    print("\nFor Q = ", q)
    print("Confusion Matrix  \n", confusion_matrix(y_test, predict))
    print("Accuracy Score  ", accuracy_score(y_test, predict))


# In[ ]:




