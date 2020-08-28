#!/usr/bin/env python
# coding: utf-8

# <h1><center> Classification of Arrhythmia</center></h1>

# **1. Importing Essential Libraries** 

# In[1]:


import pandas as pd
import numpy as np
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


# **5. generating final dataset**

# In[20]:


#Adding column names

final_df_columns=["Age","Sex","Height","Weight","QRS_Dur",
"P-R_Int","Q-T_Int","T_Int","P_Int","QRS","T","P","J","Heart_Rate",
"Q_Wave","R_Wave","S_Wave","R'_Wave","S'_Wave","Int_Def","Rag_R_Nom",
"Diph_R_Nom","Rag_P_Nom","Diph_P_Nom","Rag_T_Nom","Diph_T_Nom", 
"DII00", "DII01","DII02", "DII03", "DII04","DII05","DII06","DII07","DII08","DII09","DII10","DII11",
"DIII00","DIII01","DIII02", "DIII03", "DIII04","DIII05","DIII06","DIII07","DIII08","DIII09","DIII10","DIII11",
"AVR00","AVR01","AVR02","AVR03","AVR04","AVR05","AVR06","AVR07","AVR08","AVR09","AVR10","AVR11",
"AVL00","AVL01","AVL02","AVL03","AVL04","AVL05","AVL06","AVL07","AVL08","AVL09","AVL10","AVL11",
"AVF00","AVF01","AVF02","AVF03","AVF04","AVF05","AVF06","AVF07","AVF08","AVF09","AVF10","AVF11",
"V100","V101","V102","V103","V104","V105","V106","V107","V108","V109","V110","V111",
"V200","V201","V202","V203","V204","V205","V206","V207","V208","V209","V210","V211",
"V300","V301","V302","V303","V304","V305","V306","V307","V308","V309","V310","V311",
"V400","V401","V402","V403","V404","V405","V406","V407","V408","V409","V410","V411",
"V500","V501","V502","V503","V504","V505","V506","V507","V508","V509","V510","V511",
"V600","V601","V602","V603","V604","V605","V606","V607","V608","V609","V610","V611",
"JJ_Wave","Amp_Q_Wave","Amp_R_Wave","Amp_S_Wave","R_Prime_Wave","S_Prime_Wave","P_Wave","T_Wave",
"QRSA","QRSTA","DII170","DII171","DII172","DII173","DII174","DII175","DII176","DII177","DII178","DII179",
"DIII180","DIII181","DIII182","DIII183","DIII184","DIII185","DIII186","DIII187","DIII188","DIII189",
"AVR190","AVR191","AVR192","AVR193","AVR194","AVR195","AVR196","AVR197","AVR198","AVR199",
"AVL200","AVL201","AVL202","AVL203","AVL204","AVL205","AVL206","AVL207","AVL208","AVL209",
"AVF210","AVF211","AVF212","AVF213","AVF214","AVF215","AVF216","AVF217","AVF218","AVF219",
"V1220","V1221","V1222","V1223","V1224","V1225","V1226","V1227","V1228","V1229",
"V2230","V2231","V2232","V2233","V2234","V2235","V2236","V2237","V2238","V2239",
"V3240","V3241","V3242","V3243","V3244","V3245","V3246","V3247","V3248","V3249",
"V4250","V4251","V4252","V4253","V4254","V4255","V4256","V4257","V4258","V4259",
"V5260","V5261","V5262","V5263","V5264","V5265","V5266","V5267","V5268","V5269",
"V6270","V6271","V6272","V6273","V6274","V6275","V6276","V6277","V6278","V6279"]


# In[21]:


final_df = new_df.drop(columns = 279)
final_df.head()


# In[22]:


final_df.columns = final_df_columns
final_df.head()

