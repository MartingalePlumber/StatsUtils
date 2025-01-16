import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression

# regression toolss
def regression(df, xCols, yCol, verbose=False):
    model = LinearRegression()
    x = df[xCols]
    y = df[yCol]
    x = np.array(x)
    y = np.array(y)
    model.fit(x,y)
    coeffs = list(zip(xCols, model.coef_))
    if verbose:
        print("for regression of ", xCols, "against", yCol)
        for (n,c) in coeffs:
            print("Regression coefficient for " ,n, "against", yCol, ":", c)
        print()
    return model.coef_

def group_by_quantiles(df,name, name2, quantiles_number):
    df_new = df[[name,name2]]
    quantiles = pd.qcut(df_new[name], quantiles_number, labels=False)
    df_q = df_new.assign(quantile=quantiles.values)
    return list(df_q.groupby("quantile"))

def eliminat_quantiles(df,name, q):
    q_low = df[name].quantile(q)
    q_high = df[name].quantile(1-q)
    return df[(q_low < df[name]) & (df[name] < q_high)]

#l0 = group_by_quantiles(df, pred_name, resp_name, 100)
#l1 = [eliminat_quantiles(df0, pred_name,0.2) for (q,df0) in l0]
#df_filtered = pd.concat(l1)
#print(len(df_filtered))
#BasicStats.regression(df_filtered, [pred_name], resp_name, verbose=True)
