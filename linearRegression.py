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

''' Outliers skew the value of regression coefficients. Following distance measures are used to identify outliers
    1. Z Score
    2. Mahalanobis Distance
    3. Cook's Distance
    4. Leverage Values
    '''
# Z Score is the standardized distance of an observation from its mean value.
# Z = (X - Mu)/Sigma

from scipy.stats import zscore;
mba_salary_df['z_score_salary'] = zscore(mba_salary_df['Salary']);

#Cook's Distance

mba_influence = mba_salary_lm.get_influence();
(c, p) = mba_influence.cooks_distance;

print(np.round(c,3));

plt.stem(np.arange(len(train_X)), np.round(c,3), markerfmt=",");
plt.title("Cook's Distance Plot");
plt.xlabel("Row Index");
plt.ylabel("Cook's Distance");
plt.savefig("cooks_distance_plot.png");
plt.close();

#mba_salary_df['cooks_d'] = c;
print(mba_salary_df[11:18]);

#Leverage values of an observation measures the influence of that observation on the overall fit of the regression function and is related to the Mahalnobis distance. leverage value of more than 3(k+1)/n is treated as highly influential observation.

from statsmodels.graphics.regressionplots import influence_plot;
fig, ax = plt.subplots(figsize=(8,6));
influence_plot(mba_salary_lm, ax=ax);
plt.title("Leverage Value vs Residuals Plot");
plt.savefig("leverageVsResiduals.png");
plt.close();

# Making prediticions using the model and measuring its accuracy
pred_Y = mba_salary_lm.predict(test_X);

from sklearn.metrics import mean_squared_error, r2_score;
print("The R-Squared for test data is - ",np.abs(r2_score(test_Y, pred_Y)));

print("The mean squared error for test data is - ",np.sqrt(mean_squared_error(test_Y, pred_Y)));

#Calculating the prediction intervals
from statsmodels.sandbox.regression.predstd import wls_prediction_std;

pred_Y = mba_salary_lm.predict(test_X);
_, pred_y_low, pred_y_high = wls_prediction_std(mba_salary_lm, test_X, alpha=0.05);

pred_y_df = pd.DataFrame({'grade_10_pct': test_X['Percentage in Grade 10'],
        'actual_salary': test_Y,
        'pred_Y': pred_Y,
        'pred_Y_low': pred_y_low,
        'pred_Y_high': pred_y_high});
print(pred_y_df.head(10));

plt.scatter(pred_y_df['grade_10_pct'], pred_y_df['actual_salary'], color='blue');
plt.scatter(pred_y_df['grade_10_pct'], pred_y_df['pred_Y'], color='red');
plt.title("Actual vs Predicted Salary");
plt.xlabel("Percentage in Grade 10");
plt.ylabel("Salary");
plt.savefig("actualVsPredictedSalary.png");
plt.close();


    
    




