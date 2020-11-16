# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:50:24 2020

@author: DIY
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sms

import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.tools.tools import add_constant

work_path="C:\\Users\\DIY\\Desktop\\QM_python\\data_cw_1\\coursework_1_data_2019.csv"
save_path="C:\\Users\\DIY\\Desktop\\QM_python\\data_cw_1\\Residual vs. Fitted Plot of Increment_percent_CO.jpg"

file=pd.read_csv(work_path)

file_processed=file.drop('local_authority_area', axis=1)
file_numerical=pd.get_dummies(file_processed)
file_num_proce=file_numerical.drop(['region_Yorkshire and the Humber','local_authority_type_unitary_authority'], axis=1)


def drop_column_using_vif_(df, thresh=5):
    '''
    Calculates VIF each feature in a pandas dataframe, and repeatedly drop the columns with the highest VIF
    A constant must be added to variance_inflation_factor or the results will be incorrect

    :param df: the pandas dataframe containing only the predictor features, not the response variable
    :param thresh: the max VIF value before the feature is removed from the dataframe
    :return: dataframe with multicollinear features removed
    '''
    while True:
        # adding a constatnt item to the data
        df_with_const = add_constant(df)

        vif_df = pd.Series([variance_inflation_factor(df_with_const.values, i) 
               for i in range(df_with_const.shape[1])], name= "VIF",
              index=df_with_const.columns).to_frame()

        # drop the const
        vif_df = vif_df.drop('const')
        
        # if the largest VIF is above the thresh, remove a variable with the largest VIF
        if vif_df.VIF.max() > thresh:
            # If there are multiple variables with the maximum VIF, choose the first one
            index_to_drop = vif_df.index[vif_df.VIF == vif_df.VIF.max()].tolist()[0]
            print('Dropping: {}'.format(index_to_drop))
            df = df.drop(columns = index_to_drop)
        else:
            # No VIF is above threshold. Exit the loop
            break

    return df


file_new=drop_column_using_vif_(file_num_proce)


##############################

# If there are errors importing the data, you can also copy it in as follows:
# e.g. data = [[737.4776314, 34, 65],
#              [869.2063792, 57, 73],
#              [1033.705248, 59, 100],
#              ...
#              [737.5129466, 66, 49]]
# (Compare this example with the file demo_multreg_data_example.csv)

# These lines extract the y-values and the x-values from the data:
file_new=file_new.drop(['region_North East','region_North West','local_authority_type_non_metropolitan_county'], axis=1)
y_values = file_new.iloc[:,1]
x_values = file_new.iloc[:,4:]

# These lines perform the regression procedure:
X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
# and print a summary of the results:
print(regression_model_b.summary())
print() # blank line

# Now we store all the relevant values:
predictor_coeffs  = regression_model_b.params[1:]
constant          = regression_model_b.params[0] # called the 'intercept' in simple regression
Rsquared          = regression_model_b.rsquared
MSE               = regression_model_b.mse_resid
pvalues_T         = regression_model_b.pvalues[1:]
pvalue_F          = regression_model_b.f_pvalue

# Note that predictor_coeffs is a list of the best-fit coefficients for x1, x2, x3, ...
# Similarly, pvalues_T is a list of the p-values associated with each of these variables in turn.

# Print these summary stats:
print("predictor coefficients =\n", predictor_coeffs)
print("constant               =", constant)
print("Rsquared               =", Rsquared)
print("MSE                    =", MSE)
print("T-test pvalues         =\n", pvalues_T)
print("F-test pvalue          =", pvalue_F)

font1={'size':14}
font2={'size':14.5}
# plot 
plt.axhline(c="r",ls="--")
plt.scatter(regression_model_b.fittedvalues, regression_model_b.resid)
# adding title and labels
plt.xlabel('Fitted increment of percentage of COinE 08-18',font2)
plt.ylabel('Residual',font2)
plt.title('Residual vs. Fitted Plot of Increment_percent_CO',font1)
plt.savefig(save_path)
plt.show()

#############################

resid=list(regression_model_b.resid)
outliers=regression_model_b.resid.min()
print(resid.index(outliers))


