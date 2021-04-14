import pandas as pd

data=pd.read_csv("Stocks.csv")

Nifty100={"Symbol":[],"Id":[]}

for i in range(0,100):
    
    Nifty100["Symbol"].append(data.loc[i,"Symbol"])
    Nifty100["Id"].append(i+1)

"""   


NiftyNext_50_symbol_with_id={"Symbol":[],"Id":[]}

for i in range(50,100):
    
    NiftyNext_50_symbol_with_id["Symbol"].append(data.loc[i,"Symbol"])
    NiftyNext_50_symbol_with_id["Id"].append(i+1)
"""
print(Nifty100)
"""
#print(NiftyNext_50_symbol_with_id)
import pandas as pd
from datetime import datetime,date
from datetime import timedelta
import numpy as np
from findiff import FinDiff 
from nsepy import get_history
import json 
import mysql.connector

to_date=datetime.now()-timedelta(days=2)
to_date=datetime.strftime(to_date,'%Y,%m,%d')
to_date=datetime.strptime(to_date,'%Y,%m,%d')
from_date=to_date - timedelta(days=60)

#print(to_date,from_date)

data = get_history(symbol="SBIN", start=date(2020,1,1), end=date(2021,1,18))

#data=get_history(symbol="SBIN",start=from_date,end=to_date)
print(data)
"""