#!/usr/bin/env python
# coding: utf-8

# # Diabetes prediction with KNN¶
# 

# In[10]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,StandardScaler # for standardization
from sklearn.model_selection import train_test_split, GridSearchCV ,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score, mean_squared_error, r2_score,classification_report,roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier##  library for KNN


# In[2]:


# turn off alerts
from warnings import filterwarnings
filterwarnings ('ignore')


# # Dataset and Story

# Purpose: There is some information about the people in the data set kept in our hospital. We are asked to perform a estimation model about whether the person has diabetes according to the results of the analysis.

# In[3]:


df= pd.read_csv("diabetes.csv")


# In[4]:


df.head()


# # Model and Prediction

# In[5]:


df["Outcome"].value_counts()  # representation numbers of the dependent variable.


# Veride 1 yani şeker hastası sayısında 268 adet kişinin bilgileri, 0 yani şeker hastası olmayan kişilerin verilerinden ise 500 kişinin bilgileri bulunmaktadır.

# In[6]:


df.describe().T # descriptive statistics


# In[7]:


y=df["Outcome"]# get dependent variable
X=df.drop(["Outcome"], axis=1) # bağımsız değişkenleri alınması
X_train,X_test,y_train,y_test = train_test_split(X,# independent variable
                                                y, #the dependent variable
                                                test_size=0.30,# test data
                                                random_state=42) 


# In[11]:


knn_model=KNeighborsClassifier().fit(X,y)# model installed


# In[17]:


y_pred = knn_model.predict(X) # predictive acquisition values


# In[18]:


accuracy_score(y,y_pred) # success rate


# In[19]:


print(classification_report(y,y_pred)) #detailed reporting


# In[20]:


knn_model.predict_proba(X)[0:10]# gives the probability of classes.


# ### ROC Eğrisi

# In[33]:


logit_roc_auc = roc_auc_score(y,knn_model.predict(X)) # grafic
fpr,tpr,theresholds= roc_curve(y,knn_model.predict_proba(X)[:,1])#curve
plt.figure() 
plt.plot(fpr,tpr,label='AUC (area= %0.2f)'  % logit_roc_auc)
plt.plot([0,1],[0,1],'r--')# axis
plt.xlim([0.0,1.0])#axis
plt.ylim([0.0,1.05])#axis
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.legend ('Log_ROC')
plt.show()


# Comment: A graph that plots False-Positive rejection vs. True-Positive rejects to predict model success.

# # Model Tuning

# In[21]:


knn= KNeighborsClassifier()# model object


# In[23]:


knn_params={"n_neighbors": np.arange(1,50)}#grouping of parameters


# In[24]:


knn_cv_model=GridSearchCV(knn,knn_params,cv=10).fit(X_train,y_train)


# In[25]:


#best model success values
knn_cv_model.best_score_


# In[26]:


#the most ideal parameters
knn_cv_model.best_params_


# In[27]:


#final model
knn_tuned= KNeighborsClassifier(n_neighbors=11).fit(X_train, y_train)


# In[31]:


y_pred=knn_tuned.predict(X_test)


# In[32]:


accuracy_score(y_test,y_pred)


# In[ ]:




