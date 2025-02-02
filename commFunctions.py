# List of common functions
#  commFunctions.py
#  Regression_py
#
#  Created by Raghunath Tripasuri on 26/01/25.
#
import pandas as pd;
import numpy as np;

import matplotlib.pyplot as plt;
import statsmodels.api as sm;
import seaborn as sn;


def get_standardized_values(vals):
    return (vals - vals.mean())/vals.std();

#Variance inflation factor for multi collinearity check
from statsmodels.stats.outliers_influence import variance_inflation_factor;

def get_vif_factors(X):
    print(X.info());
    X_matrix = X.to_numpy();
    vif = [variance_inflation_factor(X_matrix, i) for i in range(X_matrix.shape[1])];
    vif_factors = pd.DataFrame();
    vif_factors['column'] = X.columns;
    vif_factors['VIF'] = vif;
    
    return vif_factors;
    
def draw_pp_plot(model, title):
    probplot = sm.ProbPlot(model.resid, fit=True);
    plt.figure(figsize=(8,6));
    probplot.ppplot(line='45');
    plt.title(title);
    plt.savefig("/Users/raghunatht/Documents/Programming/Python/Regression_py/pp_plot_multiRegression.png");
    plt.close();
    
def plot_resid_fitted(fitted, resid, title):
    plt.scatter(get_standardized_values(fitted), get_standardized_values(resid));
    plt.title(title);
    plt.xlabel("Standardized predicted values");
    plt.ylabel("Standardized residual values");
    plt.savefig("/Users/raghunatht/Documents/Programming/Python/Regression_py/residual_plot_multiRegression.png");
    plt.close();
    
# Get significant variables
def get_significant_vars(lm):
    var_p_vals_df = pd.DataFrame(lm.pvalues);
    var_p_vals_df['vars'] = var_p_vals_df.index;
    var_p_vals_df.columns = ['pvals', 'vars'];
    #filter the column names where p-value < 0.05
    return list(var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'])


from sklearn import metrics;
def draw_cm(actual, predicted):
    cm = metrics.confusion_matrix(actual, predicted);
    sn.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Good Credit', 'Bad Credit'], yticklabels=['Good Credit', 'Bad Credit']);
    plt.ylabel('True Label');
    plt.xlabel('Predicted Label');
    plt.savefig("/Users/raghunatht/Documents/Programming/Python/Regression_py/ConfusionMatrix.png");
    plt.close();


