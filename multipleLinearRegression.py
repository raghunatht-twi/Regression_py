# Functions to implement multiple linear regression
#  multipleLinearRegression.py
#  Regression_py
#
#  Created by Raghunath Tripasuri on 26/01/25.

import pandas as pd;
import numpy as np;

import math;

import matplotlib.pyplot as plt;
import seaborn as sn;

import statsmodels.api as sm;

#load the data for IPL Acution file and print metadata
ipl_auction_df = pd.read_csv("/Users/raghunatht/Documents/Programming/Python/Regression_py/Data/IPL IMB381IPL2013.csv");
print(ipl_auction_df.info());
print(ipl_auction_df[0:5][0:4]);

#Create a regression model for IPL Auction
#Get list of all features
X_features = list(ipl_auction_df.columns);
X_features = ['AGE', 'COUNTRY', 'PLAYING ROLE','T-RUNS','T-WKTS','ODI-RUNS-S','ODI-SR-B','ODI-WKTS','ODI-SR-BL','CAPTAINCY EXP','RUNS-S','HS','AVE','SR-B','SIXERS','RUNS-C','WKTS','AVE-BL','ECON','SR-BL'];
print(X_features);

#Encoding categorical features

categorical_features = ['AGE','COUNTRY','PLAYING ROLE','CAPTAINCY EXP'];
ipl_auction_encoded_df = pd.get_dummies(ipl_auction_df[X_features], columns=categorical_features, dtype='int',drop_first=True);
print(ipl_auction_encoded_df.columns);

X_features = list(ipl_auction_encoded_df.columns);
print(ipl_auction_encoded_df.head(15));

print(X_features);

#Create the test and train data split
from sklearn.model_selection import train_test_split;

X = sm.add_constant(ipl_auction_encoded_df);
Y = ipl_auction_df['SOLD PRICE'];

train_X, test_X, train_Y, test_Y = train_test_split(X,Y,train_size=0.8, random_state=42);

# Building the model on the training data set
ipl_model_1 = sm.OLS(train_Y, train_X).fit();
print(ipl_model_1.summary2());

#Multi-collinearity check
# User Variance Inflation Factor (VIF) to check for multi-collinearity

from commFunctions import get_vif_factors;
vif_factors = get_vif_factors(X[X_features]);
print(vif_factors);

columns_with_large_vif = vif_factors[vif_factors.VIF > 4].column;
print(columns_with_large_vif);

sn.heatmap(X[columns_with_large_vif].corr(), annot=True);
plt.title("Heatmap depicting correlation between features");
plt.savefig("/Users/raghunatht/Documents/Programming/Python/Regression_py/IPLHeatmap.png");
plt.close();

#Remove the columns with large VIF
columns_to_be_removed = ['T-RUNS','T-WKTS','RUNS-S','HS','AVE','RUNS-C','SR-B','AVE-BL','ECON','ODI-SR-B','ODI-RUNS-S','AGE-2','SR-BL'];
X_new_features = list(set(X_features) - set(columns_to_be_removed));
print(X_new_features);
new_vif_factors = get_vif_factors(X[X_new_features]);
print(new_vif_factors);

#Build the model with new features
train_X = train_X[X_new_features];
ipl_model_2 = sm.OLS(train_Y, train_X).fit();
print(ipl_model_2.summary2());

#Based on the model summary, only significant features are country_ind, country_eng, sixers and captaincy_exp_1. So let's create a new model based on these features
significant_vars = ['COUNTRY_IND', 'COUNTRY_ENG', 'SIXERS', 'CAPTAINCY EXP_1'];
train_X = train_X[significant_vars];
ipl_model_3 = sm.OLS(train_Y, train_X).fit();
print(ipl_model_3.summary2());

from commFunctions import draw_pp_plot;
draw_pp_plot(ipl_model_3, "Normal P-P Plot of Regression Standardized Residuals");

# Test for Homoscedasticity
from commFunctions import plot_resid_fitted;
plot_resid_fitted(ipl_model_3.fittedvalues, ipl_model_3.resid, "Residuals Plot");

# Influence Analysis
k = train_X.shape[1];
n = train_X.shape[0];

leveage_cutoff = 3*((k+1)/n);
print("Number of variables -", k, " Number of observations -", n, " Leverage cutoff -", leveage_cutoff);

from statsmodels.graphics.regressionplots import influence_plot;
fig, ax = plt.subplots(figsize=(8,6));
influence_plot(ipl_model_3, ax=ax);
plt.title("Leverage Value vs Residuals");
plt.savefig("/Users/raghunatht/Documents/Programming/Python/Regression_py/InfluenceAnalysis_multiRegression.png");
plt.close();

print(ipl_auction_df[ipl_auction_df.index.isin([23,58,83])]);

train_X_new = train_X.drop([23,58,83], axis=0);
train_Y_new = train_Y.drop([23,58,83], axis=0);

ipl_model_4 = sm.OLS(train_Y_new, train_X_new).fit();
print(ipl_model_4.summary2());

draw_pp_plot(ipl_model_4, "Normal P-P Plot of Regression Standardized Residuals");
plot_resid_fitted(ipl_model_4.fittedvalues, ipl_model_4.resid, "Residuals Plot");

#Transforming the response variable
train_Y = np.sqrt(train_Y);
ipl_model_5 = sm.OLS(train_Y, train_X).fit();
print(ipl_model_5.summary2());

draw_pp_plot(ipl_model_5, "Normal P-P Plot of Regression Standardized Residuals");

#Make predictions using model 5 we built
pred_y = np.power(ipl_model_5.predict(test_X[train_X.columns]),2);
pred_y_df = pd.DataFrame({'pred_y':pred_y, 'test_y':test_Y,'player_country':test_X['COUNTRY_IND']});
print(test_X.info())
print(pred_y_df);
plt.scatter(pred_y_df['test_y'], pred_y_df['pred_y']);
plt.title("Prediction vs Actual - multiregression");
plt.savefig("/Users/raghunatht/Documents/Programming/Python/Regression_py/PredictionVsActual_multiRegression.png");
plt.close();
print("With Debug working")
