#!/usr/bin/env python
# coding: utf-8

# <h1><center> Classification of Arrhythmia</center></h1>

# **1. Importing Essential Libraries** 

# In[1]:


import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.impute import SimpleImputer


# **2. Reading the .CSV File**

# In[2]:


#Reading csv file

df=pd.read_csv("data/arrhythmia.csv",header=None)


# *looking at first five rows of dataset*

# In[3]:


df.head()


# *Looking at Last five rows of dataset*

# In[4]:


df.tail()


# # Data Preprocessing

# **1. Description of dataframe**

# In[5]:


#Dimension of dataset.

df.shape


# In[6]:


#concise summary of the dataframe.

df.info()


# In[7]:


#descriptive statistics of dataframe.

df.describe()


# **2. checking for null values in dataset**

# In[8]:


#Counting total Number of null values

pd.isnull(df).sum().sum()


# In[9]:


#Replacing ? with np.nan value-

df = df.replace('?', np.NaN)


# In[10]:


#final counting total number of null values in dataset

nu=pd.isnull(df).sum().sum()
nu


# **3.Visualizing  the distribution of our missing data:**

# In[11]:


pd.isnull(df).sum().plot()
plt.xlabel('Columns')
plt.ylabel('Total number of null value in each column')


# In[12]:


#Zooming in

pd.isnull(df).sum()[7:17].plot()
plt.xlabel('Columns')
plt.ylabel('Total number of null value in each column')


# In[13]:


#visualizing the exact columns of missing values

pd.isnull(df).sum()[9:16].plot(kind="bar")
plt.xlabel('Columns')
plt.ylabel('Total number of null value in each column')


# In[14]:


#Dropping the column 13 as it contains some many missing values.

df.drop(columns = 13, inplace=True)


# In[ ]:





# **4. Imputer object using the mean strategy and missing_values type for imputation**

# In[15]:


# make copy to avoid changing original data (when Imputing)

new_df = df.copy()


# In[16]:


# make new columns indicating what will be imputed

cols_with_missing = (col for col in new_df.columns 
                                 if new_df[col].isnull().any())
for col in cols_with_missing:
    new_df[col] = new_df[col].isnull()


# In[17]:


# Imputation

my_imputer = SimpleImputer()
new_df = pd.DataFrame(my_imputer.fit_transform(new_df))
new_df.columns = df.columns


# In[18]:


# imputed dataframe

new_df.head()


# In[19]:


pd.isnull(new_df).sum().sum()

