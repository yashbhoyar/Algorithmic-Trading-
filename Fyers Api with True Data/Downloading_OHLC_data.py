import csv 
import time
import pandas as pd 
from truedata_ws.websocket.TD import TD
import datetime 


with open("id.txt","r") as f:
    id=f.read()
with open("password.txt","r") as f:
    password=f.read()

td_app = TD(id, password) #td_app = TD(usname, password,) 

Nifty50=['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'INFRATEL', 'BRITANNIA',
            'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HEROMOTOCO', 
            'HINDALCO', 'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'IOC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK',
            'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SHREECEM', 'SBIN', 'SUNPHARMA',
            'TCS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL', 'ULTRACEMCO', 'WIPRO', 'ZEEL']

for i in Nifty50:
    data =  td_app.get_historic_data(i, duration='3 D', bar_size='15 min')   #this will give 1 min data for previous 3 days until the current time 
    data=pd.DataFrame(data)
    x=i+".csv"
    data.columns=["Time","Open","High","Low","Close","Volume","oi"]
    data=data[["Time","Open","High","Low","Close","Volume"]]
    data["Time"]=pd.to_datetime(data["Time"],format="%Y-%m-%d %H-%M-%S")        
    data.set_index("Time",inplace=True)
    data=data.resample("15min").agg({"Open":"first","High":max,"Low":min,"Close":"last","Volume":sum})  #converting into 10min data,anytime frame can be used
    data.reset_index(inplace=True)
    data.dropna(axis="index",how="any",inplace=True)
    

    data.to_csv(x)      #saving data to csv
