#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Q1
#Part (a)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
df=pd.read_csv("daily_covid_cases.csv")
month_yr=["Feb-20","Apr-20","Jun-20","Aug-20","Oct-20","Dec-20","Feb-21","Apr-21","Jun21","Aug-21","Ot-21"]
plt.plot(df["Date"],df["new_cases"])
plt.xticks(np.arange(1,661,60),month_yr,rotation=90)
plt.title("Lineplot---Q1(a)")
plt.ylabel("New Confirmed Cases")
plt.xlabel("Month_Year")
plt.show()


# In[10]:


#Part (b)
def lag_series(p,end):#function for lag series
    p_lag=[]
    norm_p=[]
    for i in range(p,end):
        p_lag.append(df.iloc[i][1])
    for j in range(0,end-p):
        norm_p.append(df.iloc[j][1])
    return p_lag,norm_p
lag1,norm1=lag_series(1,612)
corr1,_=pearsonr(lag1,norm1)
print("correlation value::",corr1)


# In[3]:


#part (c)
plt.scatter(norm1,lag1)
plt.xticks(rotation=90)
plt.title("scatter plot with lag 1 series")
plt.xlabel("x(t)")
plt.ylabel("x(t-1)")
plt.show()


# In[11]:


# part (d)
lag2,norm2=lag_series(2,612)#to find lagged and normal time series on the complete data
lag3,norm3=lag_series(3,612)
lag4,norm4=lag_series(4,612)
lag5,norm5=lag_series(5,612)
lag6,norm6=lag_series(6,612)
corr2,_=pearsonr(lag2,norm2)
corr3,_=pearsonr(lag3,norm3)
corr4,_=pearsonr(lag4,norm4)
corr5,_=pearsonr(lag5,norm5)
corr6,_=pearsonr(lag6,norm6)
l1=[1,2,3,4,5,6]
l2=[corr1,corr2,corr3,corr4,corr5,corr6]
plt.plot(l1,l2)
plt.xlabel("lag p value--->")
plt.ylabel("correlation value--->")


# In[5]:


# part (e)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df["new_cases"],lags=6)
plt.xlabel("lag p value")
plt.ylabel("correlation values")
plt.show()


# In[6]:


#Q2
# part(a)
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = int(np.ceil(len(X)*test_size))
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
date_test=[df.iloc[i][0] for i in range(397,612)]
date_train=[df.iloc[i][0] for i in range(0,399)]
month_yr_train=["Feb-20","Apr-20","Jun-20","Aug-20","Oct-20","Dec-20","Feb-21"]
month_yr_test=["Feb-21","Apr-21","Jun21","Aug-21","Ot-21"]
plt.plot(train)
plt.xticks(np.arange(1,397,57),month_yr_train,rotation=90)
plt.title("First wave---train")
plt.xlabel("month_year")
plt.ylabel("no. of confirmed cases")
plt.show()
plt.plot(test)
plt.title("Second wave---test")
plt.xticks(np.arange(1,214,43),month_yr_test,rotation=90)
plt.xlabel("month_year")
plt.ylabel("no. of confirmed cases")
plt.show()


# In[7]:


from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_squared_error as mse
def predict(window):# to find predictions value on testing data
    model = AR(train, lags=window) 
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model 
    if(window==5):
        print("coefficients::",coef)
    #part (b)
    #using these coefficients walk forward over time steps in test, one step each time
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1] # Add other values
        obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.
    return predictions
def mape(predictions):# to find mape in %
    return np.sum(abs(np.array(test)-np.array(predictions))/(np.array(test)))*100/215
def rmse(predictions):# to find rmse in %
    return np.sqrt(mse(test,predictions))*100/(sum(test)/215)[0]
predictions=predict(5)
plt.scatter(test,predictions)
plt.xticks(rotation=90)
plt.title("scatter plot of actual and predicted values")
plt.xlabel("actual no. of confirmed cases")
plt.ylabel("predicted no. of confirmed cases")
plt.show()
# part(ii)
plt.plot(test)
plt.plot(predictions)
plt.xticks(np.arange(1,214,43),month_yr_test,rotation=90)
plt.title("plot of actual and predicted values")
plt.xlabel("month_year")
plt.ylabel("no. of confirmed cases")
plt.legend(["actual","predicted"])
plt.show()
# part(iii)
print("RMSE%::",rmse(predictions),"%")
print("MAPE%::",mape(predictions),"%")


# In[12]:


#Q3
pre1=predict(1)
pre5=predictions
pre10=predict(10)
pre15=predict(15)
pre25=predict(25)
l=['1','5','10','15','25']
rmse_lag=[rmse(pre1),rmse(pre5),rmse(pre10),rmse(pre15),rmse(pre25)]
mape_lag=[mape(pre1),mape(pre5),mape(pre10),mape(pre15),mape(pre25)]
print("RMSE for different lag values::",rmse_lag)
print("MAPE for different lag values::",mape_lag)
plt.bar(l,rmse_lag)
plt.xlabel("lag p value---")
plt.ylabel("rmse in %")
plt.show()
plt.bar(l,mape_lag)
plt.xlabel("lag p value---")
plt.ylabel("mape in %")
plt.show()


# In[9]:


#Q4
def optimised_p(N): #to find the optimal value of p lag value
    for i in range(N):
        lag,norm=lag_series(i,N)
        corr,_=pearsonr(lag,norm)
        if(abs(corr)<(2/np.sqrt(612-N))):
            opt_p=i
            return opt_p
opt_p=optimised_p(397) # optimal p on training data
predict_opt_p=predict(opt_p) # predicted value of test data
print("RMSE%::",rmse(predict_opt_p),"%")
print("MAPE%::",mape(predict_opt_p),"%")

