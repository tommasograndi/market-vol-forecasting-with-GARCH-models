
import pandas as pd 
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


# REALIZED VOLATILITY
def realized_VOL(series):
    vol = pd.Series(series.rolling(window=22).std().shift(-21))

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


# models : dictionary with key the name of the model and value the forecast of the model (or the fit)
# realized : the observed daily volatilities
def evaluate(models : dict, realized): 

    results_df = {}

    for forecast in models.values():

        # Call sklearn function to calculate MAE
        results_df['MAE'] = mean_absolute_error(realized, forecast)

        # Call sklearn function to calculate MSE
        results_df['MSE'] = mean_squared_error(realized, forecast)


    results_df = pd.DataFrame(results_df, index = models.keys())

    return results_df

# News impact curve
def NIC(model, name : str):

    vols = model.conditional_volatility.dropna() #sigma_t
    epsi = model.resid.shift(1).dropna() #shift so we get epsilon_t-1

    merge = pd.merge(vols, epsi, left_index=True, right_index=True, how='inner')

    merge['epsi_squared'] = merge.iloc[:,1]**2
    
    ## GARCH(1,1)
    if name == 'GARCH': 
        omega = model.params[2] #constant term
        alpha = model.params[3]
        unco_var = omega / (1 + alpha) #unconditional variance
        beta = model.params[4]
        A = omega + beta * unco_var

        merge['ht'] = A + (alpha * merge['epsi_squared'])
        plt.title('News Impact Curve for GARCH(1,1)')


    ## GJR-GARCH(1,1)
    elif name == 'GJR':

        omega = model.params[2] #constant term
        alpha = model.params[3]
        unco_var = omega / (1 + alpha) #unconditional variance
        beta = model.params[5]
        gamma = model.params[4]
        A = omega + beta * unco_var

        def gjr(value):
            if value > 0:
                return (A + alpha * (value**2))
            elif value < 0:
                return (A + (alpha + gamma) * (value**2))

        # Apply the function to create the ht column
        merge['ht'] = merge.iloc[:,1].apply(gjr)

        plt.title('News Impact Curve for GJR-GARCH')
    
    elif name == 'EGARCH':
        omega = model.params[2] #constant term
        alpha = model.params[3]
        unco_var = omega / (1 + alpha) #unconditional variance
        unco_vol = np.sqrt(unco_var)
        beta = model.params[5]
        gamma = model.params[4]

        A = (unco_vol**(2*beta))*np.exp(omega - (alpha * np.sqrt(2 / np.pi)))

        def egarch(value):
            if value > 0:
                return (A * np.exp((gamma + alpha)/unco_vol * value))
            elif value < 0:
                return (A * np.exp((gamma - alpha)/unco_vol * value))

        # Apply the function to create the ht column
        merge['ht'] = merge.iloc[:,1].apply(egarch)
    

    merge.sort_values(by='resid', inplace=True)
    plt.plot(merge.iloc[:,1].values, merge['ht'].values)
    plt.xlabel('$\epsilon_{t-1}$')
    plt.ylabel('$\sigma^2$')

