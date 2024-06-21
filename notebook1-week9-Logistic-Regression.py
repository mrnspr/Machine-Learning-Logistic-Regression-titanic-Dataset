#!/usr/bin/env python
# coding: utf-8

# 
# # Logistic Regression with Python
# 
# work with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). This is a very famous data set! 
# 
# We want to predict who will be saved on the Titanic.
# 

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[49]:


train = pd.read_csv('titanic_train.csv')


# In[50]:


train.head()


# # Exploratory Data Analysis
# 
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# ## Missing Data
# 
# 

# In[51]:


train.info()


# In[52]:


train.isnull()


# In[53]:


train.isnull().sum()


# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[54]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


#  The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. 
#  Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later

# In[55]:


#sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[56]:


#sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[57]:


#sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[58]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[59]:


train['Age'].plot.hist(bins=30)


# In[60]:


sns.countplot(x='SibSp',data=train)


# In[61]:


sns.countplot(x='Parch',data=train)


# ___
# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:
# 

# In[62]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[63]:


train[train['Pclass']==1]['Age'].mean()


# In[64]:


train[train['Pclass']==1]


# In[65]:


train[train['Pclass']==1]['Age']


# In[66]:


train[train['Pclass']==1]['Age'].mean()


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[67]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 29

        else:
            return 25

    else:
        return Age


# Now apply that function!

# In[68]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# Now let's check that heat map again!

# In[69]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Great! Let's go ahead and drop the Cabin column and the row in Embarked that is NaN.

# In[70]:


train.drop('Cabin',axis=1,inplace=True)


# In[71]:


train['Embarked'].value_counts()


# In[72]:


train['Embarked'].replace(np.nan, 'S', inplace=True)


# In[73]:


train.info()


# In[74]:


train.head()


# ## Converting Categorical Features 
# 
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[75]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
pclass = pd.get_dummies(train['Pclass'],drop_first=True)


# In[76]:


sex


# In[77]:


train.drop(['PassengerId','Sex','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)


# In[78]:


train = pd.concat([train,sex,embark,pclass],axis=1)


# In[79]:


train.head()


# 
# 
# # Building a Logistic Regression model
# 
# Let's start by splitting our data into a training set and test set 
# 

# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[93]:


X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)


# ## Training and Predicting

# In[94]:


from sklearn.linear_model import LogisticRegression


# In[95]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[96]:


predictions = logmodel.predict(X_test)


# In[97]:


predictions


# In[98]:


y_test


# ## Evaluation

# In[99]:


from sklearn.metrics import confusion_matrix


# In[100]:


confusion_matrix(y_test,predictions)


# We can check precision,recall,f1-score using classification report!

# In[101]:


from sklearn.metrics import classification_report


# In[102]:


print(classification_report(y_test,predictions))

