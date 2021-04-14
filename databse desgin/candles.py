import pandas as pd
from datetime import datetime,date
from datetime import timedelta
import numpy as np
from findiff import FinDiff 
from nsepy import get_history
import json 
import mysql.connector
import talib as ta


def Candlesticks_Pattern(data):
    data["Upside/Downside Gap Three Methods"]= ta.CDLXSIDEGAP3METHODS(data["Open"],data["High"],data["Low"],data["Close"])
    data["Upside Gap Two Crows"]= ta.CDLUPSIDEGAP2CROWS(data["Open"],data["High"],data["Low"],data["Close"])
    data["Unique 3 River"]= ta.CDLUNIQUE3RIVER(data["Open"],data["High"],data["Low"],data["Close"])
    data["Tristar Pattern"] =ta.CDLTRISTAR (data["Open"],data["High"],data["Low"],data["Close"])
    data["Thrusting Pattern"]= ta.CDLTHRUSTING(data["Open"],data["High"],data["Low"],data["Close"])
    data["Tasuki Gap"]= ta.CDLTASUKIGAP(data["Open"],data["High"],data["Low"],data["Close"])
    data["Takuri (Dragonfly Doji with very long lower shadow)"]= ta.CDLTAKURI(data["Open"],data["High"],data["Low"],data["Close"])
    data["Stick Sandwich"]= ta.CDLSTICKSANDWICH(data["Open"],data["High"],data["Low"],data["Close"])
    data["Stalled Pattern"]= ta.CDLSTALLEDPATTERN(data["Open"],data["High"],data["Low"],data["Close"])
    data["Spinning Top"]= ta.CDLSPINNINGTOP(data["Open"],data["High"],data["Low"],data["Close"])
    data["Short Line Candle"]= ta.CDLSHORTLINE(data["Open"],data["High"],data["Low"],data["Close"])
    data["Shooting Star"]= ta.CDLSHOOTINGSTAR(data["Open"],data["High"],data["Low"],data["Close"])
    data["Separating Lines"]= ta.CDLSEPARATINGLINES(data["Open"],data["High"],data["Low"],data["Close"])
    data["Rising/Falling Three Methods"]= ta.CDLRISEFALL3METHODS(data["Open"],data["High"],data["Low"],data["Close"])
    data["Rickshaw Man"]= ta.CDLRICKSHAWMAN(data["Open"],data["High"],data["Low"],data["Close"])
    data["Piercing Pattern"]= ta.CDLPIERCING(data["Open"],data["High"],data["Low"],data["Close"])
    data["On-Neck Pattern"]= ta.CDLONNECK(data["Open"],data["High"],data["Low"],data["Close"])
    data["Morning Star"]= ta.CDLMORNINGSTAR(data["Open"],data["High"],data["Low"],data["Close"])
    data["Morning Doji Star"]= ta.CDLMORNINGDOJISTAR(data["Open"],data["High"],data["Low"],data["Close"])
    data["Mat Hold"]= ta.CDLMATHOLD(data["Open"],data["High"],data["Low"],data["Close"])
    data["Matching Low"]= ta.CDLMATCHINGLOW(data["Open"],data["High"],data["Low"],data["Close"])
    data["Marubozu"]= ta.CDLMARUBOZU(data["Open"],data["High"],data["Low"],data["Close"])
    data["Long Line Candle"]= ta.CDLLONGLINE(data["Open"],data["High"],data["Low"],data["Close"])
    data["Long Legged Doji"]= ta.CDLLONGLEGGEDDOJI(data["Open"],data["High"],data["Low"],data["Close"])
    data["Ladder Bottom"]= ta.CDLLADDERBOTTOM(data["Open"],data["High"],data["Low"],data["Close"])
    data["Kicking - bull/bear determined by the longer marubozu"]= ta.CDLKICKINGBYLENGTH(data["Open"],data["High"],data["Low"],data["Close"])
    data["Kicking"]= ta.CDLKICKING(data["Open"],data["High"],data["Low"],data["Close"])
    data["Inverted Hammer"]= ta.CDLINVERTEDHAMMER(data["Open"],data["High"],data["Low"],data["Close"])
    data["Identical Three Crows"]= ta.CDLIDENTICAL3CROWS(data["Open"],data["High"],data["Low"],data["Close"])
    data["Two Crows"]=ta.CDL2CROWS(data["Open"],data["High"],data["Low"],data["Close"])
    data["Three Black Crows"]=ta.CDL3BLACKCROWS(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Three Inside Up/Down"]=ta.CDL3INSIDE(data["Open"],data["High"],data["Low"],data["Close"])
    data["Three-Line Strike"]=ta.CDL3LINESTRIKE(data["Open"],data["High"],data["Low"],data["Close"])
    data["Three Outside Up/Down"]=ta.CDL3OUTSIDE(data["Open"],data["High"],data["Low"],data["Close"])
    data["Three Stars In The South"]=ta.CDL3STARSINSOUTH(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Three Advancing White Soldiers"]=ta.CDL3WHITESOLDIERS(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Abandoned Baby"]=ta.CDLABANDONEDBABY(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Advance Block"]=ta.CDLADVANCEBLOCK(data["Open"],data["High"],data["Low"],data["Close"]) 
    data["Belt-hold"]=ta.CDLBELTHOLD(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Breakaway"]=ta.CDLBREAKAWAY(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Closing Marubozu"]=ta.CDLCLOSINGMARUBOZU(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Concealing Baby Swallow"]=ta.CDLCONCEALBABYSWALL(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Counterattack"]=ta.CDLCOUNTERATTACK(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Dark Cloud Cover"]=ta.CDLDARKCLOUDCOVER(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Doji"]=ta.CDLDOJI(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Doji Star"]=ta.CDLDOJISTAR(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Dragonfly Doji"]=ta.CDLDRAGONFLYDOJI(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Engulfing Pattern"]=ta.CDLENGULFING(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Evening Doji Star"]=ta.CDLEVENINGDOJISTAR(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Evening Star"]=ta.CDLEVENINGSTAR(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Up/Down-gap side-by-side white lines"]=ta.CDLGAPSIDESIDEWHITE(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Gravestone Doji"]=ta.CDLGRAVESTONEDOJI(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Hammer"]=ta.CDLHAMMER(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Hanging Man"]=ta.CDLHANGINGMAN(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Harami Pattern"]=ta.CDLHARAMI(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Harami Cross Pattern"]=ta.CDLHARAMICROSS(data["Open"],data["High"],data["Low"],data["Close"])  
    data["High-Wave Candle"]=ta.CDLHIGHWAVE(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Hikkake Pattern"]=ta.CDLHIKKAKE(data["Open"],data["High"],data["Low"],data["Close"]) 
    data["Modified Hikkake Pattern"]=ta.CDLHIKKAKEMOD(data["Open"],data["High"],data["Low"],data["Close"])  
    data["Homing Pigeon"]=ta.CDLHOMINGPIGEON(data["Open"],data["High"],data["Low"],data["Close"])  
    
    return(data)

stock_data={'Symbol': ['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'IOC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SHREECEM', 'SBIN', 'SUNPHARMA', 'TCS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL', 'ULTRACEMCO', 'WIPRO', 'ACC', 'ABBOTINDIA', 'ADANIGREEN', 'ADANITRANS', 'ALKEM', 'AMBUJACEM', 'AUROPHARMA', 'DMART', 'BAJAJHLDNG', 'BANDHANBNK', 'BANKBARODA', 'BERGEPAINT', 'BIOCON', 'BOSCHLTD', 'CADILAHC', 'COLPAL', 'CONCOR', 'DLF', 'DABUR', 'GICRE', 'GODREJCP', 'HDFCAMC', 'HAVELLS', 'HINDPETRO', 'HINDZINC', 'ICICIGI', 'ICICIPRULI', 'IGL', 'INDUSTOWER', 'NAUKRI', 'INDIGO', 'LTI', 'LUPIN', 'MARICO', 'MOTHERSUMI', 'MUTHOOTFIN', 'NMDC', 'OFSS', 'PETRONET', 'PIDILITIND', 'PEL', 'PFC', 'PGHH', 'PNB', 'SBICARD', 'SIEMENS', 'TATACONSUM', 'TORNTPHARM', 'UBL', 'MCDOWELL-N'], 'Id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 
96, 97, 98, 99, 100]}

candles_data={"Symbol":[],"Date":[],"Candle_Name":[],"stock_id":[],"pattern":[]}
candles_data=pd.DataFrame(candles_data)

to_date=datetime.now()

to_date=datetime.strftime(to_date,'%Y,%m,%d')
to_date=datetime.strptime(to_date,'%Y,%m,%d')

to_date=to_date- timedelta(days=5)
from_date=to_date - timedelta(days=70)

for t in range(0,100):
    symbol=stock_data["Symbol"][t]
    Id=stock_data["Id"][t]
    
    data = get_history(symbol=symbol, start=from_date, end=to_date)
    data.reset_index(inplace=True)
    data=Candlesticks_Pattern(data)
    data=data[-1:]
    data.reset_index(inplace=True)
    candle_names=data.columns[16:]
     
    
    for i in candle_names:
        if(data.loc[0,i]==100):
            candles_data=candles_data.append({"Symbol":symbol,"Date":data["Date"][0],"Candle_Name":i,"stock_id":Id,"pattern":"Bullish"},ignore_index=True)
        if(data.loc[0,i]==-100):
            candles_data=candles_data.append({"Symbol":symbol,"Date":data["Date"][0],"Candle_Name":i,"stock_id":Id,"pattern":"Bearish"},ignore_index=True)  

print(candles_data)
mydb=mysql.connector.connect(host='localhost',user='root',password='Yash@mysql99',auth_plugin='mysql_native_password')
cur=mydb.cursor()

q1="USE  finixsanlabs  "
cur.execute(q1)

q2="INSERT INTO eod_candlestickpatterns (date,candleName,patternType,stock_id_id) VALUES(%s,%s,%s,%s)"

b1=[]
for i in range(len(candles_data)):
    b1.append((candles_data.loc[i,"Date"],candles_data.loc[i,"Candle_Name"],candles_data.loc[i,"pattern"],int(candles_data.loc[i,"stock_id"])))

cur.executemany(q2,b1)
mydb.commit()