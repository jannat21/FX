from matplotlib.pyplot import hlines
import mplfinance as fplt
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz
import numpy as np
from datetime import datetime, timedelta


#for adding lines to chart see https://github.com/matplotlib/mplfinance/blob/master/examples/using_lines.ipynb

def  Draw_Chart(Rates, Symbol, TimeFrame, Hlines = [],  ALines=[]):

    pd.set_option('display.max_columns', 500) # number of columns to be displayed
    pd.set_option('display.width', 1500)      # max table width to display


    fplt.plot(
        Rates,
        type='candle',
        style='charles',
        title= Symbol,
        ylabel='Price ($)',
        hlines=dict(hlines=Hlines,colors=['g','r'],linestyle='-.', linewidths= [1]),
        alines=dict(alines=ALines,colors=['r'], linewidths= [1]))

    # display data
    #print("\nDisplay dataframe with data")
    #print(rates_frame)


def CopyRates (Symbol, TimeFrame, StartDate = None, Len = None):

    selected = mt5.symbol_select( Symbol, True)
    if not selected:
        print("Failed to select " + Symbol + ", error code =", mt5.last_error())
        return None
    else:
        timezone = pytz.timezone("Etc/GMT-2")
        utc_from = StartDate
        current_time = datetime.now(tz=timezone)

        if StartDate is None :
            rates = mt5.copy_rates_from_pos(Symbol, TimeFrame, 0, Len)
        else:
            rates = mt5.copy_rates_range(Symbol, TimeFrame, utc_from, current_time)
            
        #rates = mt5.copy_rates_from(Symbol, TimeFrame, current_time, 15)


        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(rates)

        # convert time in seconds into the datetime format
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        rates_frame.index = rates_frame['time']

        return rates_frame
