import pandas as pd
import mysql.connector
from nsepy import get_history
from datetime import datetime,date
from datetime import timedelta
import numpy as np

to_date=datetime.now()
to_date=datetime.strftime(to_date,'%Y,%m,%d')
to_date=datetime.strptime(to_date,'%Y,%m,%d')

from_date=to_date - timedelta(days=400)

#exporting all the company names and symbol in stocks table:
stock=pd.read_csv(r"C:\Users\Yash\Desktop\database design\Stocks.csv")

#b1={"Open":[],"High":[],"Low":[],"Close":[],"Date":[],"Volume":[],"Stock_id":[]}
#b1=pd.DataFrame(b1)
b1=[]

for j in range(0,100):
    print(stock.loc[j,"Symbol"])
    data=get_history(stock.loc[j,"Symbol"],from_date,to_date)
    data.reset_index(inplace=True)
    
    

    for i in range(len(data)):
        b1.append((float(data.loc[i,"Open"]),float(data.loc[i,"High"]),float(data.loc[i,"Low"]),float(data.loc[i,"Close"]),float(data.loc[i,"Volume"]),data.loc[i,"Date"],j+1))
       
q2="INSERT INTO eod_prices(open,high,low,close,volume,date,stock_id_id) VALUES(%s,%s,%s,%s,%s,%s,%s)"

cur.executemany(q2,b1)
mydb.commit()