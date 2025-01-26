# List of common functions
#  commFunctions.py
#  Regression_py
#
#  Created by Raghunath Tripasuri on 26/01/25.
#

def get_standardized_values(vals):
    return (vals - vals.mean())/vals.std();
