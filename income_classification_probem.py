# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 11:58:30 2022

@author: user
"""

#CLASSIFYING PERSONAL INCOME

#####Required packages#####

# To work with dataframes
import pandas as pd

# To perform numerical operations
import numpy as np

# To visualize data 
import seaborn as sns

# To partition the data
from sklearn.model_selection import train_test_split

# Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance metrics - accuracy score & confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix

#############################################################################

# Importing data
data_income = pd.read_csv(r"C:\Users\user\Documents\projects\MSc projects (CURAJ)\classification problem for subsidy\income.csv")  

# Creating a copy of original data
data = data_income.copy()

# Exploratory data analysis:
    
#1. Getting to know the data
#2. Data preprocessing (Missing values)
#3. Cross tables and data visualization

####################3 Getting to know the data ##############################

# To check variables data type
print(data.info())

# Check for missing values
data.isnull()

print('Data columns with null values:\n',data.isnull().sum())
# No missing values !!

# Summary of numerical variables
summary_num = data.describe()
print (summary_num)

# Summary of categorical variables
summary_cate = data.describe(include = "O")
print(summary_cate)

# Frequency of each catagories
data['JobType'].value_counts()
data['occupation'].value_counts()

# Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

##### There exists '?' instead of nan

# Replacing the special character '?' with nan
data = pd.read_csv(r"C:\Users\user\Documents\projects\MSc projects (CURAJ)\classification problem for subsidy\income.csv",na_values=[" ?"])

# Data pre-processing
data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
# axis = 1 means to consider at least one column value is missing

#Points to note:
#1. Missing values in JobType = 1809
#2. Missing values in Occupation = 1816
#3. There are 1809 rows where two specific columns
#  i.e. occupation and jobtype have missing values
#4. (1816 - 1809) = 7 => You still have occupation 
#  unfilled for these 7 rows. Because, jobtype is Never worked

#Dropping all the missing values bcoz those who have Jobtype nan 
#those also have the occupation nan and the rest 7 occupation 
#we can fill with never worked which is not so relevant for our classification
data2 = data.dropna(axis=0)
data_0 = data2.copy()
 
#Here we erase the 1816 rows bcoz of missing values or by assumed the jobtype never worked

# Relationship between independent variables
correlation = data2.corr() 

###################### Cross tables and Data visualization #########################

# Extracting the column names
data2.columns

# Gender proportion table:
gender = pd.crosstab(index  =  data2["gender"], columns = 'count' , normalize = True)
print(gender)

# Gender vs Salary status:
gender_salstat = pd.crosstab(index = data2["gender"], columns = data2["SalStat"], margins = True, normalize = 'index')
print(gender_salstat)

#Frequency distribution of 'Salary status'
Salstat = sns.countplot(data2['SalStat'])
## 75% of people's salary status is <= 50000
## and 25% of people's salary status is >50000

# Histogram of Age
sns.distplot(data2['age'] , bins=10 , kde=False )
## People with age 20-45 age are high in frequency

# Box plot - Age vs Salary status
sns.boxplot('SalStat', 'age', data=data2)
data2.groupby('SalStat')['age'].median()
## people with 35-50 age are more likely to earn > 50000
## people with 25-35 age are more likely to earn <=50000

# Bar plot -Job type vs Salary status
sns.countplot(y = "JobType" , data = data2 , hue = "SalStat")
pd.crosstab(index = data2['JobType'], columns = data2['SalStat'], margins= True, normalize = 'index')
## From above table it is visible that 56% of self employed people earn more than 50000

# Bar plot -Education vs Salary status
sns.countplot(y = "EdType" , data = data2 , hue = "SalStat")
pd.crosstab(index = data2['EdType'], columns = data2['SalStat'], margins= True, normalize = 'index')
## From the above table we can say that people who have done Doctorate, Masters, Prof-school are more likely to earn above 50000

# Bar plot -Occupation vs Salary status
sns.countplot(y = "occupation" , data = data2 , hue = "SalStat")
pd.crosstab(index = data2['occupation'], columns = data2['SalStat'], margins= True, normalize = 'index')
## Those who make more than 50000 per year are more likely to work as managers and professionals

# Capital gain
sns.distplot(data2['capitalgain'] , bins=10 , kde=False )
## 92% (27611) of the capital gain is 0

# Capital loss
sns.distplot(data2['capitalloss'] , bins=10 , kde=False )
## 95% (28721) of the capital loss is 0

# Hours per week vs Salary status
sns.boxplot('SalStat', 'hoursperweek', data=data2)
## From the plot it is clearly visible that those who make more than 50000 are more likely to spend 40-50 hours per week


#################### LOGISTIC REGRESSION ###########################

# Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True) 
# Converts all categorical variables into numerical type for better calculation by dividing the columns using categorical names 


# Storing the column names
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features = list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
y = new_data['SalStat'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.4, random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)

# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix_1 = confusion_matrix(test_y, prediction)
print(confusion_matrix_1)

# Calculating the accuracy
accuracy_score_1 = accuracy_score(test_y, prediction)
print(accuracy_score_1)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())

#### LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES ####

# Reindexing the salary status names to 0,1
data_0['SalStat']=data_0['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data_0['SalStat'])

cols = ['maritalstatus','relationship','race','gender']
new_data_0 = data_0.drop(cols,axis = 1)

new_data_0=pd.get_dummies(new_data_0, drop_first=True)

# Storing the column names
columns_list_0=list(new_data_0.columns)
print(columns_list_0)

# Separating the input names from data
features_0 = list(set(columns_list_0)-set(['SalStat']))
print(features_0)

# Storing the output values in y
y_0 = new_data_0['SalStat'].values
print(y_0)

# Storing the values from input features
x_0 = new_data_0[features_0].values
print(x_0)

# Splitting the data into train and test
train_x_0,test_x_0,train_y_0,test_y_0 = train_test_split(x,y,test_size=0.3,random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x_0,train_y_0)

# Prediction from test data
prediction_0 = logistic.predict(test_x_0)
print(prediction_0)

# Confusion matrix
confusion_matrix_0 = confusion_matrix(test_y_0 , prediction_0)
print(confusion_matrix_0)

# Calculating the accuracy
accuracy_score_0 = accuracy_score(test_y_0, prediction_0)
print(accuracy_score_0) 

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y_0 != prediction_0).sum())


############### KNN #####################################

# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

# import library for plotting
import matplotlib.pyplot as plt

# Storing the k nearest neighbors classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)

# Fitting the values for X and Y
KNN_classifier.fit(train_x, train_y) 

# Prediction the test values with model
prediction_1 = KNN_classifier.predict(test_x)

# Performance metric check
confusion_matrix_2 = confusion_matrix(test_y, prediction_1)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix_2)

# Calculating the accuracy
accuracy_score_2 = accuracy_score(test_y, prediction_1)
print(accuracy_score_2)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction_1).sum())
