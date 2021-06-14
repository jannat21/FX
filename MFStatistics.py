import numpy as np
from scipy import stats

def RegresionLine(Rates):
    Ys = Rates["close"] + Rates["open"]
    Ys = Ys / 2
    Xs = np.arange(Ys.size)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(Xs,Ys)
    return [[(Rates['time'][0], intercept), (Rates['time'][-1], slope * (Ys.size - 1) + intercept)]]

def StdvLines(Rates, C = 3):
    Ys = Rates["close"] + Rates["open"]
    Ys = Ys / 2
    Xs = np.arange(Ys.size)

    slope, intercept, r_value, p_value, std_err = stats.linregress(Xs,Ys)
    dif = C * np.std(Ys)

    return [[(Rates['time'][0], intercept + dif), (Rates['time'][-1], slope * (Ys.size - 1) + intercept + dif)],
            [(Rates['time'][0], intercept - dif), (Rates['time'][-1], slope * (Ys.size - 1) + intercept - dif)]]
