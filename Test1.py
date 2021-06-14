from Helpers import *
from MFStatistics import *
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pytz

if not mt5.initialize(login=41447387, server="MetaQuotes-Demo", password="oyzdutl1"):
    print("initialize() failed, error code =", mt5.last_error())
    quit()

Symbol = "AUDJPY"
TimeFrame = mt5.TIMEFRAME_H1
timezone = pytz.timezone("Etc/GMT-2")
StartDate = datetime(2021, 6, 1, tzinfo=timezone)

Rates = CopyRates(Symbol, TimeFrame, Len = 200)

alines = RegresionLine(Rates) + StdvLines(Rates, 1.5) + StdvLines(Rates, 2.5)

Draw_Chart(Rates, Symbol, TimeFrame, ALines= alines)
