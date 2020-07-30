#importing the required packages 
import pandas as pd 
import csv 
import time
import os
import numpy

#data
data=pd.read_csv("test.csv")
data=pd.DataFrame(data)

#Moving average
def MA(data,n):
    x="MA"+str(n)
    data[x]=data["Close"].rolling(n).mean()
    return data


#Exponential Moving Averages
def EMA(data,n):
    x="EMA"+str(n)
    data[x]=data["Close"].ewm(span=n).mean()
    return data



#Relative Strenght Index
def RSI(data,n):
    change=data["Close"].diff()
    gain=change.mask(change<0,0)
    loss=change.mask(change>0,0)
    average_gain=gain.ewm(com=n-1,min_periods=n).mean()
    average_loss=loss.ewm(com=n-1,min_periods=n).mean()
    rs=abs(average_gain/average_loss)
    rsi=100-100/(1+rs)
    data["RSI"]=rsi
    return(data)


#VWAP
def VWAP(data):
    typical_price= pd.Series((data['High'] + data['Low'] + data['Close']) / 3).astype(float)  
    tpv= pd.Series(data["Volume"]*typical_price).astype(float)
    cum_vol=pd.Series(data["Volume"].cumsum()).astype(float)
    data["VWAP"]=pd.Series(tpv/cum_vol).astype(float)
    return(data)


#Bolinger_bands
def Bolinger_Bands(data,n,f):                                       #n is period and f is multiplier for std deviation
    x="Middle_Band_MA"+str(n)                                       #n is 20 and f is 2 generally
    data["std_dev_close"]=data["Close"].rolling(n).std()
    data[x]=data["Close"].rolling(n).mean()
    data["Upper_Band"]=pd.Series(data[x]+f*data["std_dev_close"])
    data["Lower_Band"]=pd.Series(data[x]-f*data["std_dev_close"])
    return(data)



#MACD
def MACD(data,a,b,c):                                              #a is period of short ema , b is of long ema and c is ema used for signal
    x="exp"+str(a)                                                 #macd id exp1-exp2
    y="exp"+str(b)                                                 #exp 3 is signal line
    z="exp"+str(c)+"_signal_line"
    data[x]=data["Close"].ewm(span=a, adjust=False).mean()
    data[y]=data["Close"].ewm(span=b, adjust=False).mean()
    data[z]=data["Close"].ewm(span=c, adjust=False).mean()
    data["MACD"] = pd.Series(data[x]-data[y])
    return(data)


#VMA
def VMA(data,n):
    x="vma"+str(n)
    data['VMA']=data["Volume"].rolling(n).mean()
    return data


#SUPERTREND returns 1 for bullish and -1 for bearish

# Calculating ATR 
def ATR(data,n):
    x="ATR"+str(n)
    data[x]=data["TR"].ewm(span=n).mean()
    return data

def SUPERTREND(data,n):
    data['tr0'] = pd.Series(abs(data["High"] - data["Low"]))
    data['tr1'] = pd.Series(abs(data["High"] - data["Close"].shift(1)))
    data['tr2'] = pd.Series(abs(data["Low"]- data["Close"].shift(1)))
    data["TR"] = pd.Series(round(data[['tr0', 'tr1', 'tr2']].max(axis=1),2))


    data=ATR(data,n)
    x="ATR"+str(n)
    data['BUB'] = pd.Series(round(((data["High"] + data["Low"]) / 2) + (2 * data[x]),2))
    data['BLB'] = pd.Series(round(((data["High"] + data["Low"]) / 2) - (2 * data[x]),2))


    # FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
    #                     THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)


    for i, row in data.iterrows():
        if i==0:
            data.loc[i,"FUB"]=0.00
        else:
            if (data.loc[i,"BUB"]<data.loc[i-1,"FUB"]) or (data.loc[i-1,"Close"]>data.loc[i-1,"FUB"]):
                data.loc[i,"FUB"]=data.loc[i,"BUB"]
            else:
                data.loc[i,"FUB"]=data.loc[i-1,"FUB"]

    # FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
    #                     THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)

    for i, row in data.iterrows():
        if i==0:
            data.loc[i,"FLB"]=0.00
        else:
            if (data.loc[i,"BLB"]>data.loc[i-1,"FLB"])|(data.loc[i-1,"Close"]<data.loc[i-1,"FLB"]):
                data.loc[i,"FLB"]=data.loc[i,"BLB"]
            else:
                data.loc[i,"FLB"]=data.loc[i-1,"FLB"]



    # SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
    #                 Current FINAL UPPERBAND
    #             ELSE
    #                 IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
    #                     Current FINAL LOWERBAND
    #                 ELSE
    #                     IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
    #                         Current FINAL LOWERBAND
    #                     ELSE
    #                         IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
    #                             Current FINAL UPPERBAND


    for i, row in data.iterrows():
        if i==0:
            data.loc[i,"ST"]=0.00
        elif(data.loc[i-1,"ST"]==data.loc[i-1,"FUB"]) and  (data.loc[i,"Close"]<=data.loc[i,"FUB"]):
            data.loc[i,"ST"]=data.loc[i,"FUB"]
        elif(data.loc[i-1,"ST"]==data.loc[i-1,"FUB"])and (data.loc[i,"Close"]>data.loc[i,"FUB"]):
            data.loc[i,"ST"]=data.loc[i,"FLB"]
        elif(data.loc[i-1,"ST"]==data.loc[i-1,"FLB"])and (data.loc[i,"Close"]>=data.loc[i,"FLB"]):
            data.loc[i,"ST"]=data.loc[i,"FLB"]
        elif(data.loc[i-1,"ST"]==data.loc[i-1,"FLB"])and (data.loc[i,"Close"]<data.loc[i,"FLB"]):
            data.loc[i,"ST"]=data.loc[i,"FUB"]

    # Buy Sell Indicator
    for i, row in data.iterrows():
        x="Supertrend"+str(n)
        if i==0:
            data[x]="NA"
        elif (data.loc[i,"ST"]<data.loc[i,"Close"]) :
            data.loc[i,x]=1
        else:
            data.loc[i,x]=-1
    data.drop(columns=['tr0','tr1', 'tr2', 'TR', 'BUB', 'BLB', 'FUB', 'FLB', 'ST'],inplace=True)
    return(data)

#calculating all the values
#calculating all simple moving averages 
data=MA(data,50)
data=MA(data,30)
data=MA(data,20)
data=MA(data,10)
data=MA(data,5)

#calculating all exponential moving averages
data=EMA(data,50)
data=EMA(data,30)
data=EMA(data,20)
data=EMA(data,10)
data=EMA(data,5)

#calculating rsi
data=RSI(data,14)

#calculating vwap
data=VWAP(data)

#calculating bollinger_bands
data=Bolinger_Bands(data,20,2)

#calculating MACD
data=MACD(data,12,26,9)

#calculating_vma
data=VMA(data,10)
    
#calculating_all_supertrend   
data=SUPERTREND(data,14)
data=SUPERTREND(data,10)
data=SUPERTREND(data,7)