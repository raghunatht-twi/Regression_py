# Linear Regression Examples
# linearRegression.py
# Regression_py
#
# Created by Raghunath Tripasuri on 26/01/25.

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import math;

import statsmodels.api as sm;

print("Linear Regression Examples");

#Set the print options for pandas
np.set_printoptions(precision=4, linewidth=100);

#Load the data for MBA Salary from file
mba_salary_df = pd.read_csv("/Users/raghunatht/Documents/Programming/Python/Regression_py/Data/MBA Salary.csv");
print(mba_salary_df.head(10));
print(mba_salary_df.info());

#Create a regression model for MBA Salary
X = sm.add_constant(mba_salary_df['Percentage in Grade 10']);
print(X.head(3));

Y = mba_salary_df['Salary'];
print(Y.head(3));

# Create the test and train data split
from sklearn.model_selection import train_test_split;

train_X, test_X, train_Y, test_Y = train_test_split(X,Y,train_size=0.8, random_state=100);

#Fitting the Model - using OLS method
mba_salary_lm = sm.OLS(train_Y, train_X).fit();
print(mba_salary_lm.params);








