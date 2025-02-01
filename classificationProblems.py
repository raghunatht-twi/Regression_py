#  Classification Problems
#  classificationProblems.py
#  Regression_py
#
#  Created by Raghunath Tripasuri on 01/02/25.
#

import pandas as pd;
import numpy as np;
import math;

import matplotlib.pyplot as plt;
import seaborn as sn;

import statsmodels.api as sm;

print("Classification Problems");

credit_df = pd.read_csv("/Users/raghunatht/Documents/Programming/Python/Regression_py/Data/German Credit Data.csv");
print(credit_df.info())
print(credit_df.iloc[0:5,1:7]);
print(credit_df.iloc[0:5,7:]);
print(credit_df.status.value_counts());

#Get the list of features and remove the dependent variable ('status')
X_features = list(credit_df.columns);
X_features.remove('status');
print(X_features);

#Encoding the categorical features
encoded_credit_df = pd.get_dummies(credit_df[X_features], drop_first=True, dtype='int');

print(list(encoded_credit_df.columns));
print(encoded_credit_df[['checkin_acc_A12','checkin_acc_A13','checkin_acc_A14']].head(5));

#Create the test and train data split
Y = credit_df.status;
X = sm.add_constant(encoded_credit_df);

from sklearn.model_selection import train_test_split;
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42);

print(Y_train.value_counts(normalize=True));

#Building the logistic regression model on the training data set
logit = sm.Logit(Y_train, X_train);
logit_model = logit.fit();

print(logit_model.summary2());

from commFunctions import get_significant_vars;

significant_vars = get_significant_vars(logit_model);
print(significant_vars);

#optimize the model
final_logit = sm.Logit(Y_train, sm.add_constant(X_train[significant_vars])).fit();

print(final_logit.summary2());

#Predict the test data set
y_pred_df = pd.DataFrame({'actual': Y_test, 'predicted_prob': final_logit.predict(sm.add_constant(X_test[significant_vars]))});

print(y_pred_df.sample(10, random_state =42));

y_pred_df['predicted'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.5 else 0);

print(y_pred_df.sample(10, random_state=42));

