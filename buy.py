#importing the required packages 
import pandas as pd 
import csv 
import time
import os
import numpy


#generating a session 


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

def Volume_check(i):
        if(data.loc[i,"VMA"]>10):
            return 1

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

def sma_20_50_crossover(i):                         #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["MA20"][i-1]<data["MA50"][i-1]
    check2=data["MA20"][i+1]>data["MA50"][i+1]
    if(check1 and check2):
        return 1

def sma_5_20_crossover(i):                           #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["MA5"][i-1]<data["MA20"][i-1]
    check2=data["MA5"][i+1]>data["MA20"][i+1]
    if(check1 and check2):
        return 1

def sma_10_30_crossover(i):                         #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["MA10"][i-1]<data["MA30"][i-1]
    check2=data["MA10"][i+1]>data["MA30"][i+1]
    if(check1 and check2):
        return 1

def ema_20_50_crossover(i):                         #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["EMA20"][i-1]<data["EMA50"][i-1]
    check2=data["EMA20"][i+1]>data["EMA50"][i+1]
    if(check1 and check2):
        return 1

def ema_5_20_crossover(i):                           #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["EMA5"][i-1]<data["EMA20"][i-1]
    check2=data["EMA5"][i+1]>data["EMA20"][i+1]
    if(check1 and check2):
        return 1

def ema_10_30_crossover(i):                         #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["EMA10"][i-1]<data["EMA30"][i-1]
    check2=data["EMA10"][i+1]>data["EMA30"][i+1]
    if(check1 and check2):
        return 1

def sma_10_price_crossover(i):                      #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["Close"][i-1]<data["MA10"][i-1]
    check2=data["Close"][i+1]>data["MA10"][i+1]
    if(check1 and check2):
        return 1

def sma_20_price_crossover(i):                      #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["Close"][i-1]<data["MA10"][i-1]
    check2=data["Close"][i+1]>data["MA10"][i+1]
    if(check1 and check2):
        return 1

def sma_50_price_crossover(i):                       #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["Close"][i-1]<data["MA50"][i-1]
    check2=data["Close"][i+1]>data["MA50"][i+1]
    if(check1 and check2):
        return 1

def ema_10_price_crossover(i):                      #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["Close"][i-1]<data["EMA10"][i-1]
    check2=data["Close"][i+1]>data["EMA10"][i+1]
    if(check1 and check2):
        return 1

def ema_20_price_crossover(i):                      #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["Close"][i-1]<data["EMA10"][i-1]
    check2=data["Close"][i+1]>data["EMA10"][i+1]
    if(check1 and check2):
        return 1

def ema_50_price_crossover(i):                       #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["Close"][i-1]<data["EMA50"][i-1]
    check2=data["Close"][i+1]>data["EMA50"][i+1]
    if(check1 and check2):
        return 1

def sma_5_10_50_price_crossover(i):                               #while calling this function use the parameter the index as "rows-2" as it will check for the current data
    check1=data["MA5"][i-1]<data["MA20"][i-1]
    check2=data["MA5"][i+1]>data["MA20"][i+1]
    check3=data["MA20"][i+1]>data["MA50"][i+1] and data["Close"][i+1]>data["MA50"][i+1]
    if(check1 and check2 and check3):
        return 1
        
def ema_5_10_50_price_crossover(i):                               #while calling this function use the paraEmeter the index as "rows-2" as it will check for the current data
    check1=data["EMA5"][i-1]<data["EMA20"][i-1]
    check2=data["EMA5"][i+1]>data["EMA20"][i+1]
    check3=data["EMA20"][i]>data["EMA50"][i] and data["Close"][i]>data["EMA50"][i]
    if(check1 and check2 and check3):
        return 1 

def vwap_sma_20_crossover(i):                                   #while calling this function use the paraEmeter the index as "rows-2" as it will check for the current data
    check1=data["VWAP"][i-1]<data["MA20"][i-1]
    check2=data["VWAP"][i+1]>data["MA20"][i+1]
    if( check1 and check2):
        return 1

def vwap_price_crossover(i):                                    #while calling this function use the paraEmeter the index as "rows-2" as it will check for the current data
    check1=data["Close"][i-1]<data["VWAP"][i-1]
    check2=data["Close"][i+1]>data["VWAP"][i+1]
    if( check1 and check2):
        return 1

def MACD_signal(i):                                             #while calling this function use the paraEmeter the index as "rows-2" as it will check for the current data
    check1=data["MACD"][i-1]<data["exp9_signal_line"][i-1]
    check2=data["MACD"][i+1]>data["exp9_signal_line"][i+1]
    if(check1 and check2):
        return 1

def Bollinger_Band_signal(i):                                   #while calling this function use the paraEmeter the index as "rows-2" as it will check for the current data
    check1=data["Close"][i-1]<data["Upper_Band"][i-1]
    check2=data["Close"][i+1]>data["Upper_Band"][i+1]
    if(check1 and check2):
        return 1

#SUPPORT PRICE range    
#the aurgument here will be for a particular candle to check its price range 
#the function support price range will be same for both vwap,ema ,sma support  
#the support price range is a small area near the lower part of candle where if any of sma, ema ,vwap are present continuously for more than 2 candles, we can say a support is created 
                                          
def support_price_range(i):                                                                                  
    shadow=min(abs(data.loc[i,"Open"]-data.loc[i,"Low"]),abs(data.loc[i,"Close"]-data.loc[i,"Low"]))

    #if the body is big enough than lower shadow 
    if(abs(data.loc[i,"Open"]-data.loc[i,"Close"])>1.5*(shadow)):      
        upper_range=min(data.loc[i,"Open"],data.loc[i,"Close"])+0.5*shadow
        lower_range=data.loc[i,"Low"]-0.5*shadow
    #if body is realtively smaller
    else:
        upper_range=min(data.loc[i,"Open"],data.loc[i,"Close"])+0.2*shadow
        lower_range=data.loc[i,"Low"]-0.2*shadow

    return(upper_range,lower_range)



#the supports works well when there is an uptrend 
#can be use to check:
#1.support break out during for selling
#2.support taken and price rise for buy

#sma support                                                  
def sma_20_support(i):              #not checking volume with it 
    c=0
    for t in range(i-4,i+1):
        upper_range,lower_range=support_price_range(t)
        if(upper_range>=data.loc[t,"MA20"] and lower_range<=data.loc[t,"MA20"]):
            c=c+1
    if(c>=2):
        return 1

#vwap support                       #not checking volume             
def vwap_support(i):
    c=0
    for t in range(i-5,i+1):
        upper_range,lower_range=support_price_range(t)
        if(upper_range>=data.loc[t,"VWAP"] and lower_range<=data.loc[t,"VWAP"]):
            c=c+1
    if(c>=2):
        return 1

#ema supoort                        #not checking volume 
def ema_20_support(i):
    c=0
    for t in range(i-4,i+1):
        upper_range,lower_range=support_price_range(t)
        if(upper_range>=data.loc[t,"EMA20"] and lower_range<=data.loc[t,"EMA20"]):
            c=c+1
    if(c>=2):
        return 1

def rsi_check():
    for i in range (rows-6,rows):
        if(data.loc[i,"RSI"] <= 30):
            return 1


#for checking whether a session is bullish or bearish
def is_bearish(i):
    return(data["Open"][i]>data["Close"][i])
    
def is_bullish(i):
    return(data["Open"][i]<data["Close"][i])

def is_open_eqaul_close(i):
    return(data["Open"][i]==data["Close"][i])

#gap up and gap down 
def gap_down(i):                                               #gap_down
    
    if(is_bearish(i-1) and is_bullish(i)):               #  1 bearish bulish
        if(data["Close"][i-1]>data["Close"][i]):         #  2 bearish bearish
            return (1)                                   #  3 bullish bearish
    if(is_bearish(i-1) and is_bearish(i)):               
        if(data["Close"][i-1]>data["Open"][i]):
            return (2)
    if (is_bullish(i-1) and is_bearish(i)):
        if(data["Open"][i-1]>data["Open"][i]):
            return (3)
         
            
def gap_up(i):                                               
    if(is_bullish(i-1) and is_bullish(i)):                  #gap_down
        if(data["Close"][i-1] <data["Open"][i]):        #1.Bullish bullish
            return (1)                                  #2.Bullish Bearish
    if(is_bullish(i-1) and is_bearish(i)):              #3.Bearish bullish
        if(data["Close"][i]>data["Close"][i-1]):
            return(2)
    if(is_bearish(i-1) and is_bullish(i)):
        if(data["Open"][i-1]<data["Open"][i]):
            return(3)

#Candles
#Two candles pattern:

#  1.the peircing pattern
#{
    #previous candle bearish
    #current candle bullish
    #gapdown1 is current candle opening at lower than previous days close
    #gapdown2 is current candle oprning at lower than previous days low
    #driving condition is that the green candle closes more then halfway up the red candle 
#}
def is_peircing(i):                                              #each time the argument is the final candle's index that completes the pattern
    previous=is_bearish(i-1)
    current=is_bullish(i)
    
    gap_down1=data["Open"][i]<data["Close"][i-1]
    gap_down2=data["Open"][i]<data["Low"][i-1]
    def peircing_driving_condition(i):
        open_price_of_previous_bearish_session=data["Open"][i-1]
        halfway_price_of_previous_bearish_session=data["Close"][i-1]+((data["Open"][i-1]-data["Close"][i-1])/2)
        close_price_of_current_bullish_session=data["Close"][i]
        
        if(open_price_of_previous_bearish_session>=close_price_of_current_bullish_session>= halfway_price_of_previous_bearish_session):
            return (1)
        
        
    if(previous and current and peircing_driving_condition(i) and gap_down1): 
        return 1
    if(previous and current and peircing_driving_condition(i) and gap_down2 ): 
        return 2

#2.Bullish Harami
def is_bullish_harami(i):
    def bullish_harami_driving_condition(i):
        if(data["Close"][i-1]<data["Open"][i] and data["Open"][i-1]>data["Close"][i] ):
            return 1
    if( is_bearish(i-1) and is_bullish(i) and bullish_harami_driving_condition(i) ):
        return 1

#3.Bullish engulfing
def is_bullish_engulfing(i):
    if(data["Close"][i-1]>data["Open"][i] and data["Close"][i]>data["Open"][i-1] and is_bearish(i-1) and is_bullish(i)):
        return 1


#3 candles pattern
#this doji1 is only for the morning doji star other all types of dojis are covered separately
def is_doji1(i):
    if(abs(data["Open"][i]-data["Close"][i])<=(5*abs(data["High"][i]-data["Low"][i]))/100 ):
        return 1

#1.Morning star
def is_morning_star(i):
    def price_range(i):
        return(abs(data["Close"][i]-data["Open"][i]))
    
    pr1=price_range(i-2)
    pr2=price_range(i-1)
    close=data["Close"][i]                                  #closing of bullish session
    half_price=(data["Close"][i-2]+data["Open"][i-2])/2     #half_price_of_long_bearish_session
    
    if(is_bearish(i-2) and is_bullish(i-1) and is_bullish(i) and pr1>=2*pr2 and gap_down(i-1)==1 and gap_up(i)==1 and close>=half_price):
        return 1                        #strong_middle_candle_bullish
    if(is_bearish(i-2) and is_bearish(i-1) and is_bullish(i) and pr1>=2*pr2 and gap_down(i-1)==2 and gap_up(i)==3 and close>=half_price):
        return 2                        #average_middle_candle_bearish
    
#2.morning doji star
def is_morning_doji_star(i):
    def price_range(i):
        return(abs(data["Close"][i]-data["Open"][i]))
    
    pr1=price_range(i-2)
    pr2=price_range(i-1)
    close=data["Close"][i]                                  #closing of bullish session
    half_price=(data["Close"][i-2]+data["Open"][i-2])/2     #half_price_of_long_bearish_session
    
    if(is_bearish(i-2) and is_doji1(i-1) and is_bullish(i) and pr1>=2*pr2 and gap_down(i-1)==1 and gap_up(i)==1 and close>=half_price):
        return 1

#3.single_candles_pattern
#1.Bullish Morubozu
def is_bullish_morubozu(i):                               #here the uper and lower shadow percentage are flexible and can be changed according to need
    if(is_bullish(i)):
        body=data["Close"][i]-data["Open"][i]
        up_shadow=data["High"][i]-data["Close"][i]
        low_shadow=data["Open"][i]-data["Low"][i]
        if(up_shadow<=0.01*body and low_shadow<=0.01*body):
            return 1

#2.Weak hammer
def is_weak_hammer(i):
    current=is_bullish(i)
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if(lower_shadow>=2*body and upper_shadow<0.1):
            return 1
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if(lower_shadow>=2*body and upper_shadow<0.1):
            return 1

#3.Hammer
def is_hammer(i): 
    i=i-1
    prev=is_bearish(i-1)
    nxt=is_bullish(i+1)
    current=is_bullish(i)
        
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if( prev and nxt and (lower_shadow > 2*body) and data["Low"][i-1]<data["Close"][i]<data["Close"][i-1] and data["Low"][i-1]<data["Low"][i] and upper_shadow< 0.05*body):
            return 1
            
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if( prev and nxt and (lower_shadow >2*body) and data["Low"][i-1]<data["Open"][i]<data["Close"][i-1] and data["Low"][i-1]<data["Low"][i] and upper_shadow< 0.05*body ):
           return 1
    
#4.Strong hammer
def is_strong_hammer(i):
    i=i-1
    prev=is_bearish(i-1)
    nxt=is_bullish(i+1)
    current=is_bullish(i)
    
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        
        if(prev and nxt  and (lower_shadow >2*body) and gap_up(i+1)==1 and gap_down(i)==1 and upper_shadow<=0.01*body):
           return 1

#4.weak Inverted hammer
def is_weak_inverted_hammer(i):
    current=is_bullish(i)
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if(upper_shadow>2*body and lower_shadow<0.05*body):
            return 1
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if(upper_shadow>2*body and lower_shadow<0.05*body ):
            return 1

#5.inverted hammer
def is_inverted_hammer(i): 
    i=i-1
    prev=is_bearish(i-1)
    nxt=is_bullish(i+1)
    current=is_bullish(i)
        
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if( prev and nxt and (upper_shadow > 2*body) and  lower_shadow< 0.05*body):
            return 1
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if( prev and nxt and (upper_shadow >2*body) and  lower_shadow< 0.05*body ):
            return 1

#6.strong inverted hammer           
def is_strong_inverted_hammer(i):
    i=i-1
    prev=is_bearish(i-1)
    nxt=is_bullish(i+1)
    current=is_bullish(i)
    
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        t1=gap_up(i+1)
        t2=gap_down(i)
        if( prev and nxt and (upper_shadow>2*body) and t1 and t2 and (lower_shadow<0.03*body)):
            return 1

#7.Doji
def doji(i):
    body=abs(data["Open"][i]-data["Close"][i])
    total_candle=data["High"][i]-data["Low"][i]
    if(body<0.1*total_candle):
        if(is_bullish(i)):
            up_shadow=data["High"][i]-data["Close"][i]
            low_shadow=data["Open"][i]-data["Low"][i]
        
            if(up_shadow<0.1*low_shadow):
                #dragonfly_doji
                return 1
            elif(low_shadow<0.1*up_shadow):
                #graveyard_doji.append()
                return 2
            else:
                #common_doji.append()
                return 3
        
        if(is_bearish(i)):
            up_shadow=data["High"][i]-data["Open"][i]
            low_shadow=data["Close"][i]-data["Low"][i]
               
            if(up_shadow<0.1*low_shadow):
                #dragonfly_doji
                return 4
            elif(low_shadow<0.1*up_shadow):
                #graveyard_doji
                return 5
            else:
                #common_doji
                return 6 
                
    if(is_open_eqaul_close(i)):
        up_shadow=data["High"][i]-data["Close"][i]
        low_shadow=data["Open"][i]-data["Low"][i]
        if(data["High"][i]==data["Low"][i]):
            #price_doji.append()
            return 7
        elif(up_shadow==low_shadow):
            #perfect_doji.append()
            return 8
        elif(up_shadow<0.1*low_shadow):
            #dragonfly_doji.append()
            return 9
        elif(low_shadow<0.1*up_shadow):
            #graveyard_doji
            return 10
        else:
            #common_doji.append()
            return 11

#supertrend follwed by rsi 
def supertrend_rsi(i):
    yes=0
    for t in range (i-5,i):
        if(data.loc[t,"Supertrend14"]==-1 and data.loc[t+1,"Supertrend14"]==1):
            yes=1
    if(yes==1 and data.loc[i,"RSI"]<=30):
        return 1

#double supertrend
def double_supertrend(i):
    t1=0
    t2=0
    for t in range (i-5,i):
        if(data.loc[t,"Supertrend14"]==-1 and data.loc[t+1,"Supertrend14"]==1):
            t1=1
        if(data.loc[t,"Supertrend14"]==1 and data.loc[t+1,"Supertrend14"]==-1):
            t1=0
        if(data.loc[t,"Supertrend10"]==-1 and data.loc[t+1,"Supertrend10"]==1):
            t2=1
        if(data.loc[t,"Supertrend10"]==1 and data.loc[t+1,"Supertrend10"]==-1):
            t2=0
    
    if(t1 and t2):
        return 1

def supertrend_change(i):
    for t in range (i-5,i):
        if(data.loc[t,"Supertrend14"]==-1 and data.loc[t+1,"Supertrend14"]==1):
            return 1
        if(data.loc[t,"Supertrend10"]==-1 and data.loc[t+1,"Supertrend10"]==1):
            return 1
        if(data.loc[t,"Supertrend7"]==-1 and data.loc[t+1,"Supertrend7"]==1):
            return 1



#creating a output data frame
stock={"Symbol":[],"Strategy":[],"Entry":[],"StopLoss":[],"Target1":[],"Target2":[],"Target3":[]}
stock=pd.DataFrame(stock)

#creating a list of nifty50 stocks
#data=pd.read_csv(r"C:\Users\Yash\Downloads\ind_nifty50list.csv")
#stock_list=[]
#for i in range (0,50):
    #stock_list.append(data.loc[i,"Symbol"])

stock_list=['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'INFRATEL', 'BRITANNIA',
            'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HEROMOTOCO', 
            'HINDALCO', 'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'IOC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK',
            'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SHREECEM', 'SBIN', 'SUNPHARMA',
            'TCS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL', 'ULTRACEMCO', 'VEDL', 'WIPRO', 'ZEEL']

#checking all startegies:

for stock in stocklist:
    #getting data for i stock:
    name=stock+".csv"
    data=pd.read_csv("name")
    data=pd.DataFrame(data)
    rows,columns=data.shape

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
    
    
    #getting the levels for i stock:
    
    levels=pd.read_csv("levels.csv")
    levels.set_index("Symbol",inplace=True)
    levels_values=list(levels.loc[i,"gn720":"g720"])
    

    for t in range(0,31):
        if(data.loc[rows-1,"Close"]>=levels_values[t] and data.loc[rows-1,"Close"]<=levels_values[t+1]):
            stop_loss=levels_values[t]
            entry=levels_values[t+1]
            target1=levels_values[t+2]
            target2=levels_values[t+3]
            target3=levels_values[t+4]
            break

    #checking all startegies:
    
    if(sma_5_20_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_5_20_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(sma_10_30_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_10_30_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(sma_20_50_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_20_50_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(ema_5_20_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_5_20_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(ema_10_30_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_10_30_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(ema_20_50_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_20_50_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(sma_10_price_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_10_price_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(sma_20_price_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_20_price_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(sma_50_price_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_50_price_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(ema_10_price_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"ema_10_price_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(ema_20_price_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"ema_20_price_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(ema_50_price_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"ema_50_price_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(sma_5_10_50_price_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_5_10_50_price_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(ema_5_10_50_price_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"ema_5_10_50_price_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(vwap_sma_20_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"vwap_sma_20_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(vwap_price_crossover(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"vwap_price_crossover","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(sma_20_support(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"sma_20_support","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(ema_20_support(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"ema_20_support","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(vwap_support(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"vwap_support","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(supertrend_rsi(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"supertrend_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(double_supertrend(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"double_supertrend_change","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(supertrend_change(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"supertrend_change","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(MACD_signal(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"MACD_crosses_ema9","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if(Bollinger_Band_signal(rows-2) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"price above upper bollinger band","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    
    #from here onwards supertrend change and candles are checked for reversal :
    if((is_peircing(rows-3) or is_peircing(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"peircing_suprtrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_bullish_engulfing(rows-3) or is_bullish_engulfing(rows-4)) and  supertrend_change(rows-1)  and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"bullish_engulfing_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_bullish_harami(rows-3) or is_bullish_harami(rows-4))  and supertrend_change(rows-1) and  Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"bullish_harami_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_bullish_morubozu(rows-3) or is_bullish_morubozu(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"bullish_morubozu_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_hammer(rows-3) or is_hammer(rows-4)) and  supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"hammer_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_weak_hammer(rows-3) or is_weak_hammer(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"weak_hammer_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_strong_hammer(rows-3) or is_strong_hammer(rows-4)) and  supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"strong_hammer_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)    
    if((is_inverted_hammer(rows-3) or is_inverted_hammer(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"inverted_hammer_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_strong_inverted_hammer(rows-3) or is_strong_inverted_hammer(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"strong_inverted_hammer_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_weak_inverted_hammer(rows-3) or is_weak_inverted_hammer(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"weak_inverted_hammer_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_morning_doji_star(rows-3) or is_morning_doji_star(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"morning_doji_star_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((is_morning_star(rows-3) or is_morning_star(rows-4)) and supertrend_change(rows-1) and  Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"morning_star_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((doji(rows-3)==1 or doji(rows-4)==1)  and  supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"dragonfly_doji_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    if((doji(rows-3)==2 or  doji(rows-4))  and  supertrend_change(rows-1) and  Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"graveyard_doji_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},index=True)
    if((doji(rows-3)==3 or  doji(rows-4)) and supertrend_change(rows-1) and  Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"doji_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},index=True)
    if((doji(rows-3)==7 or  doji(rows-4))  and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"price_doji_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},index=True)
    if((doji(rows-3)==8 or  doji(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"perfectdoji__supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},index=True)
    if((doji(rows-3)==9 or  doji(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"perfect_dragonfly_doji_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},index=True)
    if((doji(rows-3)==10 or  doji(rows-4)) and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"perfect_graveyard_doji_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},index=True)
    if((doji(rows-3)==11  or  doji(rows-4))  and supertrend_change(rows-1) and Volume_check(rows-1)):
        stock=stock.append({"Symbol":i,"Strategy":"perfect_common_doji_supertrend","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},index=True)

    if(rsi_check()):
        if(is_peircing(rows-2) and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"peircing_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_bullish_engulfing(rows-2) and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"bullish_engulfing_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_bullish_harami(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"bullish_harami_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_bullish_morubozu(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"bullish_morubozu_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_hammer(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"hammer_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_weak_hammer(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"weak_hammer_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_strong_hammer(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"strong_hammer_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)    
        if(is_inverted_hammer(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"inverted_hammer_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_strong_inverted_hammer(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"strong_inverted_hammer_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_weak_inverted_hammer(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"weak_inverted_hammer_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_morning_doji_star(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"morning_doji_star_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(is_morning_star(rows-2)and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"morning_star_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(doji(rows-2)==1 and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"dragonfly_doji_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(doji(rows-2)==2 and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"graveyard_doji_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(doji(rows-2)==3 and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"doji_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(doji(rows-2)==7 and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"price_doji_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(doji(rows-2)==8 and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"perfectdoji__rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(doji(rows-2)==9 and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"perfect_dragonfly_doji_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(doji(rows-2)==10 and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"perfect_graveyard_doji_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
        if(doji(rows-2)==11 and Volume_check(rows-1)):
            stock=stock.append({"Symbol":i,"Strategy":"perfect_common_doji_rsi","Entry":entry,"StopLoss":stop_loss,"Target1":target1,"Target2":target2,"Target3":target3},ignore_index=True)
    

stock.to_csv("test.csv") 
