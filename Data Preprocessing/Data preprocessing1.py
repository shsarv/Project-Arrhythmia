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

