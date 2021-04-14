import talib as ta
import pandas as pd
from datetime import datetime

data=pd.read_csv(r"E:\Startups\Wide Eye\Main Scripts\Buy\SBIN.csv")
data["DateTime"]=data["Time"]
data["Time"]=pd.to_datetime(data["DateTime"],format="%Y-%m-%d").dt.time
data["Date"]=pd.to_datetime(data["DateTime"],format="%Y-%m-%d").dt.date
data.drop(columns=["Unnamed: 0","Unnamed: 0.1","Unnamed: 0.1.1"],inplace=True)
filt1=(data["Time"]!=datetime.strptime("09:00:00", "%H:%M:%S").time()) 
filt2=(data["Time"]!=datetime.strptime("15:30:00", "%H:%M:%S").time())
filt3=(data["Time"]!=datetime.strptime("16:00:00", "%H:%M:%S").time())
data["Time"]=data.loc[filt1,"Time"]
data["Time"]=data.loc[filt2,"Time"]
data["Time"]=data.loc[filt3,"Time"]
data.dropna(inplace=True)
data.reset_index(inplace=True)
rows,columns=data.shape
data1=data

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

data=Candlesticks_Pattern(data)
data.to_csv("Candlesticks")