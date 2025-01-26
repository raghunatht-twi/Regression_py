# Linear Regression Examples
# linearRegression.py
# Regression_py
#
# Created by Raghunath Tripasuri on 26/01/25.

import pandas as pd;
import numpy as np;
import math;

import matplotlib.pyplot as plt;
import seaborn as sn;

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

# Check for the Model Diagnostics
''' The following are used for validating teh simple linear regression models
    1. Co-efficient of determiniation (R-squared)
    2. Hypothese test for hte regression coefficients
    3. Analysis of variance for overall model validity
    4. Residual analysis to vlaidate the regression model assumptions
    5. Outlier analysis, since the presense of outliers can significantly impact the regression parameters '''
print(mba_salary_lm.summary());

'''Residual analysis:
    1. The residuals should be normally distributed
    2. Variance of residuals should be constant
    3. The functional form of regression is correctly specificed
    4. There are no outliers in the data '''
    
mba_salary_resid = mba_salary_lm.resid;
print (" MBA SUMMARY RESID -", mba_salary_resid.head(7));
probplot = sm.ProbPlot(mba_salary_resid, fit=True);
#plt.figure(figsize=(8,6));
probplot.ppplot(line='45');
plt.title("Normal P-P Plot of Regression Standardized Residuals");
plt.savefig("residual_analysis.png");
plt.close();

print(mba_salary_lm.resid);
    
#Test for Homoscedasticity
from commFunctions import get_standardized_values;
plt.scatter(get_standardized_values(mba_salary_lm.fittedvalues), get_standardized_values(mba_salary_lm.resid));
plt.title("Residual Plot");
plt.xlabel("Standardized predicted values");
plt.ylabel("Standardized Residuals");
plt.savefig("residual_plot.png");
plt.close();
    
#Outlier Analysis






