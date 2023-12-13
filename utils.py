
import pandas as pd 
import numpy as np


def realized_vol(series):
    vol = pd.Series(series.rolling(window=25).std().shift(-24))

    return vol

def rolling_vol(series, window, variance=False):

    vol = pd.Series(series.rolling(window = window).std(), index = series.index)

    return vol


def GarmanKlass(data, sd=True):
    
    # Extract the series length
    n = len(data)
    
    # Extract the adjusting coefficient
    coeff = data['Adj Close'] / data['Close'] #adjusting coefficient
    
    # Adjust the high low open close 
    H = np.log(data['High'] * coeff)
    L = np.log(data['Low'] * coeff)
    O = np.log(data['Open'] * coeff)
    C = np.log(data['Close'] * coeff)
    
    # Calculate normalized returns
    u = H - O #between high and open
    d = L - O #between low and open
    c = C - O #between close and open
    
    ## Calculate the GK VARIANCE estimator
    x = 0.511 * (u - d)**2 + (-0.019) * (c * (u + d) - 2 * u * d) + (-0.383) * c**2
    
    # return the series
    return pd.Series(np.sqrt(x), index = data.index) 