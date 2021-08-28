#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
warnings.filterwarnings("ignore")


# #  Read the data

# In[42]:


train = pd.read_csv("Traindata.csv",index_col ='ID') 
test = pd.read_csv("Testdata.csv",index_col ='ID')


# # Understanding Data

# In[43]:


#check the features present in our data
train.columns, train.shape


# We have <b>21 independent variables</b>(without ID) and <b>1 target variable</b>, i.e. Business_Sourced in the train dataset.

# In[44]:


test.columns, test.shape


# We have similar features in the test dataset as the train dataset __except the Business_Sourced__

# In[45]:


# Printing data types for each variable train.dtypes
train.dtypes


# In[46]:


train.head()


# # Univariate Analysis 

# In[47]:


#frequency table of target variable
train['Business_Sourced'].value_counts()


# In[48]:


# Normalize can be set to True to print proportions instead of number 
train['Business_Sourced'].value_counts(normalize=True)


# In[49]:


#bar plot for visulalisation of distribution of target variable
train['Business_Sourced'].value_counts().plot.bar()


# * Nearly 3000 out of 9527 are able to source business within 3 months of recruitment .

# ## Categorical Variable

# * Let's organize these categorical datas.

# In[50]:


train['Applicant_Qualification'].unique()


# In[51]:


test['Applicant_Qualification'].unique() 


# In[52]:


categorized_data = {'Applicant_Gender':{'M':1,'F':0},
                    'Manager_Gender':{'M':1,'F':0},
                    'Applicant_Marital_Status':{'S':0,'M':1,'W':2,'D':3},
                    'Applicant_Occupation':{'Others':0,'Business':1,'Salaried':2,'Self Employed':3,'Student':4},
                    'Applicant_Qualification':{'Others':0,'Class XII':1,'Class X':2,'Graduate':3,
                                               'Masters of Business Administration':4,
                                               'Associate / Fellow of Institute of Chartered Accountans of India':5,
                                               'Associate/Fellow of Institute of Company Secretories of India':6,
                                               'Associate/Fellow of Acturial Society of India':7,
                                               'Certified Associateship of Indian Institute of Bankers':8,
                                               'Associate/Fellow of Insurance Institute of India':9,
                                               'Professional Qualification in Marketing':10,
                                               'Associate/Fellow of Institute of Institute of Costs and Works Accountants of India':11},
                                                                         
                    'Manager_Joining_Designation':{'Other':0,'Level 1':1,'Level 2':2,'Level 3':3,
                                                   'Level 4':4,'Level 5':5,'Level 6':6,'Level 7':7},
                    'Manager_Current_Designation':{'Other':0,'Level 1':1,'Level 2':2,'Level 3':3,
                                                   'Level 4':4,'Level 5':5},
                    
                    'Manager_Status':{'Probation':0,'Confirmation':1}
             }
                                               


# In[53]:


train.replace(categorized_data, inplace=True)


# In[54]:


test.replace(categorized_data, inplace=True)


# ### Applicants' Status

# In[55]:


plt.figure(1) 
plt.subplot(221) 
train['Applicant_Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Applicant_Gender') 
plt.subplot(222)
train['Applicant_Qualification'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Applicant_Qualification')#ordinal variable
plt.subplot(223) 
train['Applicant_Marital_Status'].value_counts(normalize=True).plot.bar(title= 'Applicant_Marital_Status') 
plt.subplot(224)
train['Applicant_Occupation'].value_counts(normalize=True).plot.bar(title= 'Applicant_Occupation') 
plt.show()


# ### Managers' status

# In[56]:


plt.figure(2) 
plt.subplot(221)
train['Manager_Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Manager_Gender')
plt.subplot(222) 
train['Manager_Joining_Designation'].value_counts(normalize=True).plot.bar(title= 'Manager_Joining_Designation')
plt.subplot(223) 
train['Manager_Current_Designation'].value_counts(normalize=True).plot.bar(title= 'Manager_Current_Designation')
plt.subplot(224) 
train['Manager_Status'].value_counts(normalize=True).plot.bar(title= 'Manager_Status')
plt.show()


# ## Numerical Variable

# In[57]:


plt.figure(1) 
plt.subplot(121) 
sns.distplot(train['Manager_Num_Application']); 
plt.subplot(122) 
train['Manager_Num_Application'].plot.box(figsize=(16,5)) 
plt.show()


# In[58]:


plt.figure(2) 
plt.subplot(121) 
sns.distplot(train['Manager_Num_Coded']); 
plt.subplot(122) 
train['Manager_Num_Coded'].plot.box(figsize=(16,5)) 
plt.show()


# In[59]:


plt.figure(3) 
plt.subplot(121) 
df=train.dropna() 
sns.distplot(df['Manager_Business']); 
plt.subplot(122) 
df['Manager_Business'].plot.box(figsize=(16,5)) 
plt.show()


# In[60]:


plt.figure(3) 
plt.subplot(121) 
df=train.dropna() 
sns.distplot(df['Manager_Business2']); 
plt.subplot(122) 
df['Manager_Business2'].plot.box(figsize=(16,5)) 
plt.show()


# In[61]:


plt.figure(5) 
plt.subplot(121) 
sns.distplot(train['Manager_Num_Products']); 
plt.subplot(122) 
train['Manager_Num_Products'].plot.box(figsize=(16,5)) 
plt.show()


# In[62]:


plt.figure(6) 
plt.subplot(121) 
sns.distplot(train['Manager_Num_Products2']); 
plt.subplot(122) 
train['Manager_Num_Products2'].plot.box(figsize=(16,5)) 
plt.show()


# # Bivariate Analysis

# In[63]:


Applicant_Gender = pd.crosstab(train['Applicant_Gender'],train['Business_Sourced'])
Applicant_Qualification = pd.crosstab(train['Applicant_Qualification'],train['Business_Sourced']) 
Applicant_Marital_Status = pd.crosstab(train['Applicant_Marital_Status'],train['Business_Sourced']) 
Applicant_Occupation = pd.crosstab(train['Applicant_Occupation'],train['Business_Sourced'])
Applicant_Gender.div(Applicant_Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()
Applicant_Qualification.div(Applicant_Qualification.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 
Applicant_Marital_Status.div(Applicant_Marital_Status.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show() 
Applicant_Occupation.div(Applicant_Occupation.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 


# * It can be inferred that the proportion of male and female applicants is more or less __same__ for able to source business within 3 months of recruitment.
# * In applicant's qualification 6,7 group was not able to business sourced, and group of 8,9,10 has every applicant was able to business source.
# * The proportion of Single and Married applicants were able to business sourced more than Widow and Divorced.
# * In applicant occupation none of the self_employed applicant's able to business sourced.

# In[64]:


Manager_Gender = pd.crosstab(train['Manager_Gender'],train['Business_Sourced']) 
Manager_Joining_Designation = pd.crosstab(train['Manager_Joining_Designation'],train['Business_Sourced']) 
Manager_Current_Designation = pd.crosstab(train['Manager_Current_Designation'],train['Business_Sourced'])
Manager_Status = pd.crosstab(train['Manager_Status'],train['Business_Sourced']) 
Manager_Gender.div(Manager_Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 
Manager_Joining_Designation.div(Manager_Joining_Designation.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show() 
Manager_Current_Designation.div(Manager_Current_Designation.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 
Manager_Status.div(Manager_Status.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show()


# * The proportion of male and female managers is more or less __same__ was able to source business within 3 months of recruitment.
# * Not a single Level 5 manager was able to business sourced.
# * The proportion of Confirmation and Probation managers is more or less __same__ was able to source business within 3 months of recruitment.

# In[65]:


# heat map to visualize the correlation
matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu")


# # Missing Values and Outlier Treatment 

# In[66]:


train.isnull().sum()


# * For numerical variables: imputation using mean or median
# * For categorical variables: imputation using mode

# In[67]:


train['Applicant_Gender'].fillna(train['Applicant_Gender'].mode()[0], inplace=True)
train['Applicant_BirthDate'].fillna(train['Applicant_BirthDate'].mode()[0], inplace=True) 
train['Applicant_Marital_Status'].fillna(train['Applicant_Marital_Status'].mode()[0], inplace=True) 
train['Applicant_Occupation'].fillna(train['Applicant_Occupation'].mode()[0], inplace=True)
train['Applicant_Qualification'].fillna(train['Applicant_Qualification'].mode()[0], inplace=True)
train['Manager_DOJ'].fillna(train['Manager_DOJ'].mode()[0], inplace=True)
train['Manager_Joining_Designation'].fillna(train['Manager_Joining_Designation'].mode()[0], inplace=True)
train['Manager_Current_Designation'].fillna(train['Manager_Current_Designation'].mode()[0], inplace=True)
train['Manager_Grade'].fillna(train['Manager_Grade'].mode()[0], inplace=True)
train['Manager_DoB'].fillna(train['Manager_DoB'].mode()[0], inplace=True)
train['Manager_Num_Application'].fillna(train['Manager_Num_Application'].mode()[0], inplace=True)
train['Manager_Status'].fillna(train['Manager_Status'].mode()[0], inplace=True)
train['Manager_Gender'].fillna(train['Manager_Gender'].mode()[0], inplace=True)
train['Manager_Num_Coded'].fillna(train['Manager_Num_Coded'].mode()[0], inplace=True)
train['Manager_Business'].fillna(train['Manager_Business'].mode()[0], inplace=True)
train['Manager_Num_Products'].fillna(train['Manager_Num_Products'].mode()[0], inplace=True)
train['Manager_Business2'].fillna(train['Manager_Business2'].mode()[0], inplace=True)
train['Manager_Num_Products2'].fillna(train['Manager_Num_Products2'].mode()[0], inplace=True)
train['Business_Sourced'].fillna(train['Business_Sourced'].mode()[0], inplace=True)
train['Applicant_City_PIN'].fillna(train['Applicant_City_PIN'].mode()[0], inplace=True)


# In[68]:


train.isnull().sum()


# In[69]:


train.loc[train['Manager_Business']>650000,'Manager_Business']=np.mean(train['Manager_Business'])


# In[70]:


train.loc[train['Manager_Business2']>500000,'Manager_Business2']=np.mean(train['Manager_Business2'])


# In[71]:


x_train=train.drop(['Business_Sourced','Application_Receipt_Date','Applicant_BirthDate','Manager_DOJ','Manager_DoB'],axis=1)
y_train=train['Business_Sourced']

test = test.drop(['Application_Receipt_Date','Applicant_BirthDate','Manager_DOJ','Manager_DoB'],axis =1)


# In[72]:


x_train.head()


# In[73]:


test.head()


# # Model Building

# In[74]:


test.fillna(x_train.mean(), inplace=True)


# In[75]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(min_samples_split=170)
classifier.fit(x_train,y_train)


# In[76]:


prediction = classifier.predict(test)


# In[77]:


classifier.score(x_train,y_train)


# In[78]:


res={'ID':test.index,'Business_Sourced':prediction}
output=pd.DataFrame(res)


# In[79]:


output['Business_Sourced'].value_counts()


# In[80]:


output.to_csv('submission_data.csv',index=False)

