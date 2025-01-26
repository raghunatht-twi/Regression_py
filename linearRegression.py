# Linear Regression Examples
# linearRegression.py
# Regression_py
#
# Created by Raghunath Tripasuri on 26/01/25.

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import math;

print("Linear Regression Examples");

#Set the print options for pandas
np.set_printoptions(precision=4, linewidth=100);

#Load the data for MBA Salary from file
mba_salary_df = pd.read_csv("/Users/raghunatht/Documents/Programming/Python/Regression_py/Data/MBA Salary.csv");
print(mba_salary_df.head(10));
print(mba_salary_df.info());



