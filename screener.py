import pandas as pd
from datetime import datetime,date
from datetime import timedelta
import numpy as np
from findiff import FinDiff 
from nsepy import get_history
import json 
import mysql.connector

to_date=datetime.now()-timedelta(days=1)
to_date=datetime.strftime(to_date,'%Y,%m,%d')
to_date=datetime.strptime(to_date,'%Y,%m,%d')

from_date=to_date - timedelta(days=60)

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
    data['x']=data["Volume"].rolling(n).mean()
    return data


def EMA_ST(df, base, target, period, alpha=False):
    """
    Function to compute Exponential Moving Average (EMA)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the EMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)
        
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])
    
    if (alpha == True):
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()
    
    df[target].fillna(0, inplace=True)
    return df

def ATR(df, period, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute Average True Range (ATR)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (TR)
            ATR (ATR_$period)
    """
    
    atr = 'ATR_' + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'TR' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())
        
        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)
        
        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

    # Compute EMA of true range using ATR formula after ignoring first row
    EMA_ST(df, 'TR', atr, period, alpha=True)
    
    return df

def SuperTrend(df, period, multiplier, ohlc=['Open', 'High', 'Low', 'Close']):

    ATR(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)
    
    """
    SuperTrend Algorithm :
    
        BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
        BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR
        
        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                            THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)
        
        SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                            Current FINAL LOWERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                    Current FINAL UPPERBAND
    """
    
    # Compute basic upper and lower bands
    df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
    df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]
    
    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] <= df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] >  df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] <  df['final_lb'].iat[i] else 0.00 
                
    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), '0',  '1'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop([ 'TR','basic_ub', 'basic_lb', 'final_ub', 'final_lb',], inplace=True, axis=1)
    
    df.fillna(0, inplace=True)

    return df

def Calculation_Of_Techincal_Indicators(data):
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



    #calculating bollinger_bands
    data=Bolinger_Bands(data,20,2)

    #calculating MACD
    data=MACD(data,12,26,9)

    #calculating_vma
    data=VMA(data,10)

    #supertrend
    data=SuperTrend(data, 14, 3, ohlc=['Open', 'High', 'Low', 'Close'])
    data=SuperTrend(data, 10, 3, ohlc=['Open', 'High', 'Low', 'Close'])
    data=SuperTrend(data, 7, 3, ohlc=['Open', 'High', 'Low', 'Close'])

    return data

def EOD_buy():

    def sma_20_50_crossover(i):                         #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["MA20"][i-1]<data["MA50"][i-1]
        check2=data["MA20"][i]>data["MA50"][i]
        if(check1 and check2):
            return 1

    def sma_5_20_crossover(i):                           #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["MA5"][i-1]<data["MA20"][i-1]
        check2=data["MA5"][i]>data["MA20"][i]
        if(check1 and check2):
            return 1
        
    def sma_10_30_crossover(i):                           #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["MA10"][i-1]<data["MA30"][i-1]
        check2=data["MA10"][i]>data["MA30"][i]
        if(check1 and check2):
            return 1


    def ema_20_50_crossover(i):                         #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["EMA20"][i-1]<data["EMA50"][i-1]
        check2=data["EMA20"][i]>data["EMA50"][i]
        if(check1 and check2):
            return 1

    def ema_5_20_crossover(i):                           #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["EMA5"][i-1]<data["EMA20"][i-1]
        check2=data["EMA5"][i]>data["EMA20"][i]
        if(check1 and check2):
            return 1

    def ema_10_30_crossover(i):                           #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["EMA10"][i-1]<data["EMA30"][i-1]
        check2=data["EMA10"][i]>data["EMA30"][i]
        if(check1 and check2):
            return 1

    def sma_10_price_crossover(i):                      #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["Close"][i-1]<data["MA10"][i-1]
        check2=data["Close"][i]>data["MA10"][i]
        if(check1 and check2):
            return 1

    def sma_20_price_crossover(i):                       #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["Close"][i-1]<data["MA20"][i-1]
        check2=data["Close"][i]>data["MA20"][i]
        if(check1 and check2):
            return 1

    def sma_50_price_crossover(i):                       #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["Close"][i-1]<data["MA50"][i-1]
        check2=data["Close"][i]>data["MA50"][i]
        if(check1 and check2):
            return 1

    def ema_10_price_crossover(i):                      #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["Close"][i-1]<data["EMA10"][i-1]
        check2=data["Close"][i]>data["EMA10"][i]
        if(check1 and check2):
            return 1

    def ema_20_price_crossover(i):                       #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["Close"][i-1]<data["MA20"][i-1]
        check2=data["Close"][i]>data["MA20"][i]
        if(check1 and check2):
            return 1

    def ema_50_price_crossover(i):                       #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["Close"][i-1]<data["EMA50"][i-1]
        check2=data["Close"][i]>data["EMA50"][i]
        if(check1 and check2):
            return 1
    def sma_5_10_50_price_crossover(i):                               #while calling this function use the parameter the index as "i-2" as it will check for the current data
        check1=data["MA5"][i-1]<data["MA10"][i-1]
        check2=data["MA5"][i]>data["MA10"][i]
        check3=data["MA10"][i]>data["MA50"][i] and data["Close"][i]>data["MA50"][i]
        if(check1 and check2 and check3):
            return 1
            
    def ema_5_10_50_price_crossover(i):                               #while calling this function use the paraEmeter the index as "i-2" as it will check for the current data
        check1=data["EMA5"][i-1]<data["EMA10"][i-1]
        check2=data["EMA5"][i]>data["EMA10"][i]
        check3=data["EMA10"][i]>data["EMA50"][i] and data["Close"][i]>data["EMA50"][i]
        if(check1 and check2 and check3):
            return 1 


    def rsi_oversold(i):
        if(data["RSI"][i]<=30):
            return 1

    #double supertrend
    def triple_supertrend(i):
        t1=0
        t2=0
        t3=0
        for t in range (i-5,i+1):
            if(data.loc[t,"STX_14_3"]==-1 and data.loc[t+1,"STX_14_3"]==1):
                t1=1
            if(data.loc[t,"STX_14_3"]==1 and data.loc[t+1,"STX_14_3"]==-1):
                t1=0
            if(data.loc[t,"STX_10_3"]==-1 and data.loc[t+1,"STX_10_3"]==1):
                t2=1
            if(data.loc[t,"STX_10_3"]==1 and data.loc[t+1,"STX_10_3"]==-1):
                t2=0
            if(data.loc[t,"STX_7_3"]==-1 and data.loc[t+1,"STX_7_3"]==1):
                t3=1
            if(data.loc[t,"STX_7_3"]==1 and data.loc[t+1,"STX_7_3"]==-1):
                t3=0
        
        if(t1 and t2 and t3):
            return 1

    def supertrend_change(i):
        
        if(data.loc[i-1,"STX_14_3"]==-1 and data.loc[i,"STX_14_3"]==1):
            return 14
        if(data.loc[i-1,"STX_10_3"]==-1 and data.loc[i,"STX_10_3"]==1):
            return 10


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
            upper_range=data.loc[i,"Low"]+0.1*shadow
            lower_range=data.loc[i,"Low"]-0.1*shadow

        return(upper_range,lower_range)



    #the supports works well when there is an uptrend 
    #can be use to check:
    #1.support break out during for selling
    #2.support taken and price rise for buy

    #sma support                                                  
    def sma_20_support(i):              #not checking volume with it 
        c1=0
        c2=0
        for t in range(i-4,i+1):
            upper_range,lower_range=support_price_range(t)
            if(upper_range>=data.loc[t,"MA20"] and lower_range<=data.loc[t,"MA20"]):
                c1=c1+1
        for t in range(i-1,i+1):
            upper_range,lower_range=support_price_range(t)
            if(upper_range>=data.loc[t,"MA20"] and lower_range<=data.loc[t,"MA20"]):
                c2=c2+1
        if(c1>=2 and c2>=1):
            return 1

    #ema support                        #not checking volume 
    def ema_20_support(i):
        c1=0
        c2=0
        for t in range(i-4,i+1):
            upper_range,lower_range=support_price_range(t)
            if(upper_range>=data.loc[t,"EMA20"] and lower_range<=data.loc[t,"EMA20"]):
                c1=c1+1
        for t in range(i-1,i+1):
            upper_range,lower_range=support_price_range(t)
            if(upper_range>=data.loc[t,"EMA20"] and lower_range<=data.loc[t,"EMA20"]):
                c2=c2+1
        if(c1>=2 and c2>=1):
            return 1
        



    def MACD_signal(i):                                             #while calling this function use the paraEmeter the index as "i-2" as it will check for the current data
        check1=data["MACD"][i-1]<data["exp9_signal_line"][i-1]
        check2=data["MACD"][i]>data["exp9_signal_line"][i]
        if(check1 and check2):
            return 1

    def Bollinger_Band_signal(i):                                   #while calling this function use the paraEmeter the index as "i-2" as it will check for the current data
        check1=data["Close"][i-1]<data["Upper_Band"][i-1]
        check2=data["Close"][i]>data["Upper_Band"][i]
        if(check1 and check2):
            return 1


    def Bollinger_Band_RSI(i):
        check1=(data.loc[i,"Close"]>=data.loc[i,"Upper_Band"])
        check2=(data.loc[i,"RSI"]>=60 and data.loc[i,"RSI"]<=70)
        if(check1 and check2):
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
            if(data["Close"][i-1]<data["Low"][i] and data["Open"][i-1]>data["High"][i] ):
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
        if(body<0.03*total_candle):
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

    def Inside_Candle(data):
        rows1,columns1=data.shape
        mother_candles=[]
        upper=[]
        lower=[]
        no_of_inside_candles=0
        mother_candle_index=0
        for i in range (1,rows1-1):
            #checking for the patterns 
            check1=(data.loc[i-1,"High"] > data.loc[i,"High"])
            check2=(data.loc[i-1,"Low"] < data.loc[i,"Low"])
            if (check1 and check2 ):
                mother_candles.append(i-1)
                upper.append(data.loc[i-1,"High"])
                lower.append(data.loc[i-1,"Low"])
                no_of_inside_candles=+1
                    
                    
                    
            #checking for the breakouts
            for j in range(0,no_of_inside_candles) :   
                check1=(data.loc[i-1,"Close"] < upper[j]  and data.loc[i,"Close"] >upper[j]) #Upper breakout
                check2=(data.loc[i-1,"Close"] > lower[j]  and data.loc[i,"Close"] <lower[j]) #lower breakout
                mother_candle_index=mother_candles[j]
                if(check1):
                    return 1
                    

    #creating a output data frame
    stock={"Symbol":[],"Strategy":[],"Stock_id":[],"Strategy":[]}
    stock=pd.DataFrame(stock)

    #creating a list of Nifty100 stocks
    #data=pd.read_csv(r"C:\Users\Yash\Downloads\ind_Nifty100list.csv")
    #stock_list=[]
    #for i in range (0,50):
        #stock_list.append(data.loc[i,"Symbol"])

    """Nifty100=['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BRITANNIA',
                'CIPLA', 'COALINDIA', 'DRREDDY', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HEROMOTOCO', 
                'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY',  'KOTAKBANK',
                'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'RELIANCE', 'SBIN', 'SUNPHARMA',
                'TCS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL',  'WIPRO']"""

    Nifty100={
        'Symbol': ['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'IOC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SHREECEM', 'SBIN', 'SUNPHARMA', 'TCS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL', 'ULTRACEMCO', 'WIPRO', 'ACC', 'ABBOTINDIA', 'ADANIGREEN', 'ADANITRANS', 'ALKEM', 'AMBUJACEM', 'AUROPHARMA', 'DMART', 'BAJAJHLDNG', 'BANDHANBNK', 'BANKBARODA', 'BERGEPAINT', 'BIOCON', 'BOSCHLTD', 'CADILAHC', 'COLPAL', 'CONCOR', 'DLF', 'DABUR', 'GICRE', 'GODREJCP', 'HDFCAMC', 'HAVELLS', 'HINDPETRO', 'HINDZINC', 'ICICIGI', 'ICICIPRULI', 'IGL', 'INDUSTOWER', 'NAUKRI', 'INDIGO', 'LTI', 'LUPIN', 'MARICO', 'MOTHERSUMI', 'MUTHOOTFIN', 'NMDC', 'OFSS', 'PETRONET', 'PIDILITIND', 'PEL', 'PFC', 'PGHH', 'PNB', 'SBICARD', 'SIEMENS', 'TATACONSUM', 'TORNTPHARM', 'UBL', 'MCDOWELL-N'], 
    
        'Id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 
        96, 97, 98, 99, 100]
    }
                
    gan_static_levels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21, 23, 25, 28, 31, 34, 37, 40, 43, 46, 49, 53, 57, 61, 65, 69, 73, 77, 81, 86, 91, 96, 101, 106, 111, 116, 121, 127, 133, 139, 145, 151, 157, 163, 169, 176, 183, 190, 197, 204, 211, 218, 225, 233, 241, 249, 257, 265, 273, 281, 289, 298, 307, 316, 325, 334, 343, 352, 361, 371, 381, 391, 401, 411, 421, 431, 441, 452, 463, 474, 485, 496, 507, 518, 529, 541, 553, 565, 577, 589, 601, 613, 625, 638, 651, 664, 677, 690, 703, 716, 729, 743, 757, 771, 785, 799, 813, 827, 841, 856, 871, 886, 901, 916, 931, 946, 961, 977, 993, 1009, 1025, 1041, 1057, 1073, 1089, 1106, 1123, 1140, 1157, 1174, 1191, 1208, 1225, 1243, 1261, 1279, 1297, 1315, 1333, 1351, 1369, 1388, 1407, 1426, 1445, 1464, 1483, 1502, 1521, 1541, 1561, 1581, 1601, 1621, 1641, 1661, 1681, 1702, 1723, 1744, 1765, 1786, 1807, 1828, 1849, 1871, 1893, 1915, 1937, 1959, 1981, 2003, 2025, 2048, 2071, 2094, 2117, 2140, 2163, 2186, 2209, 2233, 2257, 2281, 2305, 2329, 2353, 2377, 2401, 2426, 2451, 2476, 2501, 2526, 2551, 2576, 2601, 2627, 2653, 2679, 2705, 2731, 2757, 2783, 2809, 2836, 2863, 2890, 2917, 2944, 2971, 2998, 3025, 3053, 3081, 3109, 3137, 3165, 3193, 3221, 3249, 3278, 3307, 3336, 3365, 3394, 3423, 3452, 3481, 3511, 3541, 3571, 3601, 3631, 3661, 3691, 3721, 3752, 3783, 3814, 3845, 3876, 3907, 3938, 3969, 4001, 4033, 4065, 4097, 4129, 4161, 4193, 4225, 4258, 4291, 4324, 4357, 4390, 4423, 4456, 4489, 4523, 4557, 4591, 4625, 4659, 4693, 4727, 4761, 4796, 4831, 4866, 4901, 4936, 4971, 5006, 5041, 5077, 5113, 5149, 5185, 5221, 5257, 5293, 5329, 5366, 5403, 5440, 5477, 5514, 5551, 5588, 5625, 5663, 5701, 5739, 5777, 5815, 5853, 5891, 5929, 5968, 6007, 6046, 6085, 6124, 6163, 6202, 6241, 6281, 6321, 6361, 6401, 6441, 6481, 6521, 6561, 6602, 6643, 6684, 6725, 6766, 6807, 6848, 6889, 6931, 6973, 7015, 7057, 7099, 7141, 7183, 7225, 7268, 7311, 7354, 7397, 7440, 7483, 7526, 7569, 7613, 7657, 7701, 7745, 7789, 7833, 7877, 7921, 7966, 8011, 8056, 8101, 8146, 8191, 8236, 8281, 8327, 8373, 8419, 8465, 8511, 8557, 8603, 8649, 8696, 8743, 8790, 8837, 8884, 8931, 8978, 9025, 9073, 9121, 9169, 9217, 9265, 9313, 9361, 9409, 9458, 9507, 9556, 9605, 9654, 9703, 9752, 9801, 9851, 9901, 9951, 10001, 10051, 10101, 10151, 10201, 10252, 10303, 10354, 10405, 10456, 10507, 10558, 10609, 10661, 10713, 10765, 10817, 10869, 10921, 10973, 11025, 11078, 11131, 11184, 11237, 11290, 11343, 11396, 11449, 11503, 11557, 11611, 11665, 11719, 11773, 11827, 11881, 11936, 11991, 12046, 12101, 12156, 12211, 12266, 12321, 12377, 12433, 12489, 12545, 12601, 12657, 12713, 12769, 12826, 12883, 12940, 12997, 13054, 13111, 13168, 13225, 13283, 13341, 13399, 13457, 13515, 13573, 13631, 13689, 13748, 13807, 13866, 13925, 13984, 14043, 14102, 14161, 14221, 14281, 14341, 14401, 14461, 14521, 14581, 14641, 14702, 14763, 14824, 14885, 14946, 15007, 15068, 15129, 15191, 15253, 15315, 15377, 15439, 15501, 15563, 15625, 15688, 15751, 15814, 15877, 15940, 16003, 16066, 16129, 16193, 16257, 16321, 16385, 16449, 16513, 16577, 16641, 16706, 16771, 16836, 16901, 16966, 17031, 17096, 17161, 17227, 17293, 17359, 17425, 17491, 17557, 17623, 17689, 17756, 17823, 17890, 17957, 18024, 18091, 18158, 18225, 18293, 18361, 18429, 18497, 18565, 18633, 18701, 18769, 18838, 18907, 18976, 19045, 19114, 19183, 19252, 19321, 19391, 19461, 19531, 19601, 19671, 19741, 19811, 19881, 19952, 20023, 20094, 20165, 20236, 20307, 20378, 20449, 20521, 20593, 20665, 20737, 20809, 20881, 20953, 21025, 21098, 21171, 21244, 21317, 21390, 21463, 21536, 21609, 21683, 21757, 21831, 21905, 21979, 22053, 22127, 22201, 22276, 22351, 22426, 22501, 22576, 22651, 22726, 22801, 22877, 22953, 23029, 23105, 23181, 23257, 23333, 23409, 23486, 23563, 23640, 23717, 23794, 23871, 23948, 24025, 24103, 24181, 24259, 24337, 24415, 24493, 24571, 24649, 24728, 24807, 24886, 24965, 25044, 25123, 25202, 25281, 25361, 25441, 25521, 25601, 25681, 25761, 25841, 25921, 26002, 26083, 26164, 26245, 26326, 26407, 26488, 26569, 26651, 26733, 26815, 26897, 26979, 27061, 27143, 27225, 27308, 27391, 27474, 27557, 27640, 27723, 27806, 27889, 27973, 28057, 28141, 28225, 28309, 28393, 28477, 28561, 28646, 28731, 28816, 28901, 28986, 29071, 29156, 29241, 29327, 29413, 29499, 29585, 29671, 29757, 29843, 29929, 30016, 30103, 30190, 30277, 30364, 30451, 30538, 30625, 30713, 30801, 30889, 30977, 31065, 31153, 31241, 31329, 31418, 31507, 31596, 31685, 31774, 31863, 31952, 32041, 32131, 32221, 32311, 32401, 32491, 32581, 32671, 32761, 32852, 32943, 33034, 33125, 33216, 33307, 33398, 33489, 33581, 33673, 33765, 33857, 33949, 34041, 34133, 34225, 34318, 34411, 34504, 34597, 34690, 34783, 34876, 34969, 35063, 35157, 35251, 35345, 35439, 35533, 35627, 35721, 35816, 35911, 36006, 36101, 36196, 36291, 36386, 36481, 36577, 36673, 36769, 36865, 36961, 37057, 37153, 37249, 37346, 37443, 37540, 37637, 37734, 37831, 37928, 38025, 38123, 38221, 38319, 38417, 38515, 38613, 38711, 38809]

    for i in range(0,50):
        
        #td_app = TD('FYERS452', 'jxVM5P54')
        #data = td_app.get_historic_data(i, duration='3 M', bar_size='15 min')
        #symbol=i+".csv"
        #data=pd.read_csv(symbol)
        
        #data=pd.DataFrame(data)
        #data.columns=["Time","Open","High","Low","Close","Volume","oi"]
        #data=data[["Time","Open","High","Low","Close","Volume"]]


        #data.set_index("Time",inplace=True)
        #data=data.resample("1D").agg({"Open":"first","High":max,"Low":min,"Close":"last","Volume":sum})
        #data.dropna(inplace=True)
        #data.reset_index(inplace=True)
        #rows,columns=data.shape

        
        
        data = get_history(symbol=Nifty100["Symbol"][i], start=date(2020,10,1), end=date(2021,4,6))
        data.reset_index(inplace=True)
        print(data)
        data=Calculation_Of_Techincal_Indicators(data)
        rows,column=data.shape

        print(Nifty100["Symbol"][i])
        """
        for t in range(len(gan_static_levels)):
            if(data.loc[rows-1,"Close"]>=gan_static_levels[t] and data.loc[rows-1,"Close"]<=gan_static_levels[t+1]):
                stop_loss=gan_static_levels[t]
                entry=gan_static_levels[t+1]
                target1=gan_static_levels[t+2]
                target2=gan_static_levels[t+3]
                target3=gan_static_levels[t+4]
                break """

        #checking all startegies:
        
        if(sma_5_20_crossover(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-5-20 crossover"},ignore_index=True)
        if(sma_10_30_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-10-30 crossover"},ignore_index=True)
        if(sma_20_50_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-20-50 crossover"},ignore_index=True)
        if(ema_5_20_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-5-20 crossover"},ignore_index=True)
        if(ema_20_50_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-20-50 crossover"},ignore_index=True)
        if(ema_10_30_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-10-30 crossover"},ignore_index=True)
        if(sma_20_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-20 pricecrossover"},ignore_index=True)
        if(sma_50_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-50 pricecrossover"},ignore_index=True)
        if(ema_20_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-20 pricecrossover"},ignore_index=True)
        if(ema_50_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-50 pricecrossover"},ignore_index=True)
        if(sma_5_10_50_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-5-10-50 pricecrossover"},ignore_index=True)
        if(ema_5_10_50_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-5-10-50 pricecrossover"},ignore_index=True)
        
        if(sma_20_support(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-20 support"},ignore_index=True)
        if(ema_20_support(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-20 support"},ignore_index=True)
        
        if(Bollinger_Band_RSI(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"BollingerBand-RSI"},ignore_index=True)
        
            
        if(MACD_signal(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"MACD"},ignore_index=True)
        if(Bollinger_Band_signal(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"BollingerBand"},ignore_index=True)
        if(rsi_oversold(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"RSI-oversold"},ignore_index=True)
        if(triple_supertrend(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"TripleSupertrendChange-14-10-7"},ignore_index=True)
        if(supertrend_change(rows-1)==14):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SuperTrend change-14"},ignore_index=True)
        if(supertrend_change(rows-1)==10):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SuperTrend change-10"},ignore_index=True)
        
        if(is_peircing(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Peircing"},ignore_index=True)
        if(is_bullish_engulfing(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Bullish-Engulfing"},ignore_index=True)
        if(is_bullish_harami(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Bullish-Harami"},ignore_index=True)
        if(is_bullish_morubozu(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Bullish-Morubozu"},ignore_index=True)
        if(is_hammer(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Hammer"},ignore_index=True)
        if(is_weak_hammer(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Weak-Hammer"},ignore_index=True)
        if(is_strong_hammer(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Strong-Hammer"},ignore_index=True)    
        if(is_inverted_hammer(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Inverted-Hammer"},ignore_index=True)
        if(is_strong_inverted_hammer(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Strong-Inverted-Hammer"},ignore_index=True)
        if(is_weak_inverted_hammer(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Weak-Inverted-Hammer"},ignore_index=True)
        if(is_morning_doji_star(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Morning-Doji-Star"},ignore_index=True)
        if(is_morning_star(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Morning-Star"},ignore_index=True)
        if(doji(rows-1)==1 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Dragonfly-Doji"},ignore_index=True)
        if(doji(rows-1)==2 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Graveyard-Doji"},ignore_index=True)
        if(doji(rows-1)==3 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Doji"},ignore_index=True)
        if(doji(rows-1)==7 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Price-Doji"},ignore_index=True)
        if(doji(rows-1)==8 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Perfect-Doji"},ignore_index=True)
        if(doji(rows-1)==9 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Perfect-Dragonfly-Doji"},ignore_index=True)
        if(doji(rows-1)==10 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Perfect-Graveyard-Doji"},ignore_index=True)
        if(doji(rows-1)==11 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Perfect-Common-Doji"},ignore_index=True)


    return(stock)


def sell_EOD():

    def sma_20_50_crossover(i):                         #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["MA20"][i-1]>data["MA50"][i-1]
        check2=data["MA20"][i]<data["MA50"][i]
        if(check1 and check2):
            return 1

    def sma_5_20_crossover(i):                           #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["MA5"][i-1]>data["MA20"][i-1]
        check2=data["MA5"][i]<data["MA20"][i]
        if(check1 and check2):
            return 1

    def sma_10_30_crossover(i):                         #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["MA10"][i-1]>data["MA30"][i-1]
        check2=data["MA10"][i]<data["MA30"][i]
        if(check1 and check2):
            return 1

    def ema_20_50_crossover(i):                         #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["EMA20"][i-1]>data["EMA50"][i-1]
        check2=data["EMA20"][i]<data["EMA50"][i]
        if(check1 and check2):
            return 1

    def ema_5_20_crossover(i):                           #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["EMA5"][i-1]>data["EMA20"][i-1]
        check2=data["EMA5"][i]<data["EMA20"][i]
        if(check1 and check2):
            return 1

    def ema_10_30_crossover(i):                         #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["EMA10"][i-1]>data["EMA30"][i-1]
        check2=data["EMA10"][i]<data["EMA30"][i]
        if(check1 and check2):
            return 1

    def sma_10_price_crossover(i):                      #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["Close"][i-1]>data["MA10"][i-1]
        check2=data["Close"][i]<data["MA10"][i]
        if(check1 and check2):
            return 1

    def sma_20_price_crossover(i):                      #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["Close"][i-1]>data["MA10"][i-1]
        check2=data["Close"][i]<data["MA10"][i]
        if(check1 and check2):
            return 1

    def sma_50_price_crossover(i):                       #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["Close"][i-1]>data["MA50"][i-1]
        check2=data["Close"][i]<data["MA50"][i]
        if(check1 and check2):
            return 1

    def ema_10_price_crossover(i):                      #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["Close"][i-1]>data["EMA10"][i-1]
        check2=data["Close"][i]<data["EMA10"][i]
        if(check1 and check2):
            return 1

    def ema_20_price_crossover(i):                      #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["Close"][i-1]>data["EMA10"][i-1]
        check2=data["Close"][i]<data["EMA10"][i]
        if(check1 and check2):
            return 1

    def ema_50_price_crossover(i):                       #while calling this function use the parameter the index as "rows-1" as it will check for the current data
        check1=data["Close"][i-1]>data["EMA50"][i-1]
        check2=data["Close"][i]<data["EMA50"][i]
        if(check1 and check2):
            return 1

    def vwap_sma_20_crossover(i):                                   #while calling this function use the paraEmeter the index as "rows-1" as it will check for the current data
        check1=data["VWAP"][i-1]>data["MA20"][i-1]
        check2=data["VWAP"][i]<data["MA20"][i]
        if( check1 and check2):
            return 1

    def vwap_price_crossover(i):                                    #while calling this function use the paraEmeter the index as "rows-1" as it will check for the current data
        check1=data["Close"][i-1]>data["VWAP"][i-1]
        check2=data["Close"][i]<data["VWAP"][i]
        if( check1 and check2):
            return 1

    def MACD_signal(i):                                             #while calling this function use the paraEmeter the index as "rows-1" as it will check for the current data
        check1=data["MACD"][i-1]>data["exp9_signal_line"][i-1]
        check2=data["MACD"][i]<data["exp9_signal_line"][i]
        if(check1 and check2):
            return 1

    def Bollinger_Band_signal(i):                                   #while calling this function use the paraEmeter the index as "rows-1" as it will check for the current data
        check1=data["Close"][i-1]>data["Lower_Band"][i-1]
        check2=data["Close"][i]<data["Lower_Band"][i]
        if(check1 and check2):
            return 1

    #double supertrend
    def triple_supertrend(i):
        t1=0
        t2=0
        t3=0
        for t in range (i-5,i+1):
            if(data.loc[t,"STX_14_3"]==1 and data.loc[t+1,"STX_14_3"]==-1):
                t1=1
            if(data.loc[t,"STX_14_3"]==-1 and data.loc[t+1,"STX_14_3"]==1):
                t1=0
            if(data.loc[t,"STX_10_3"]==1 and data.loc[t+1,"STX_10_3"]==-1):
                t2=1
            if(data.loc[t,"STX_10_3"]==-1 and data.loc[t+1,"STX_10_3"]==1):
                t2=0
            if(data.loc[t,"STX_7_3"]==1 and data.loc[t+1,"STX_7_3"]==-1):
                t3=1
            if(data.loc[t,"STX_7_3"]==-1 and data.loc[t+1,"STX_7_3"]==1):
                t3=0
        
        if(t1 and t2 and t3):
            return 1

    def supertrend_change(i):
        
        if(data.loc[i-1,"STX_14_3"]==1 and data.loc[i,"STX_14_3"]==-1):
            return 14
        if(data.loc[i-1,"STX_10_3"]==1 and data.loc[i,"STX_10_3"]==-1):
            return 10


    def rsi_overbought(i):
        if(data["RSI"][i]>=70):
            return 1

    #RESISTANCE PRICE RANGE 
    #the aurgument here will be for a particular candle to check its price range 
    #the function resistance price range will be same for both vwap,ema ,sma resistance 
    #the resistance price range is a small area near the lower part of candle where if any of sma, ema ,vwap are present continuously for more than 2 candles, we can say a resistance is created
    def resistance_price_range(i):                                                                                  
        shadow=min(abs(data.loc[i,"High"]-data.loc[i,"Open"]),abs(data.loc[i,"High"]-data.loc[i,"Close"]))

        #if the body is big enough than lower shadow 
        if(abs(data.loc[i,"Open"]-data.loc[i,"Close"])>1.5*(shadow)):      
            upper_range=data.loc[i,"High"]+0.5*shadow
            lower_range=max(data.loc[i,"Open"],data.loc[i,"Close"])-0.5*shadow
            
        #if body is realtively smaller
        else:
            upper_range=data.loc[i,"High"]+0.2*shadow
            lower_range=max(data.loc[i,"Open"],data.loc[i,"Close"])-0.2*shadow

        return(upper_range,lower_range)

    #the resistance works well when there is an downtrend 
    #can be use to check:
    #1.resistance  break out during for buying
    #2.resistance taken and price fall for sell
    #sma resistance                                                  
    def sma_20_resistance(i):
        c1=0
        c2=0
        for t in range(i-4,i+1):
            upper_range,lower_range=resistance_price_range(t)
            if(upper_range>=data.loc[t,"MA20"] and lower_range<=data.loc[t,"MA20"]):
                c1=c1+1
        for t in range(i-1,i+1):
            upper_range,lower_range=resistance_price_range(t)
            if(upper_range>=data.loc[t,"MA20"] and lower_range<=data.loc[t,"MA20"]):
                c2=c2+1
        if(c1>=2 and c2>=1):
            return 1

    def ema_20_resistance(i):
        c1=0
        c2=0
        for t in range(i-4,i+1):
            upper_range,lower_range=resistance_price_range(t)
            if(upper_range>=data.loc[t,"EMA20"] and lower_range<=data.loc[t,"EMA20"]):
                c1=c1+1
        for t in range(i-1,i+1):
            upper_range,lower_range=resistance_price_range(t)
            if(upper_range>=data.loc[t,"EMA20"] and lower_range<=data.loc[t,"EMA20"]):
                c2=c2+1
        if(c1>=2 and c2>=1):
            return 1

    #vwap resistance 
    def vwap_resistance(i):
        c1=0
        c2=0
        for t in range(i-4,i+1):
            upper_range,lower_range=resistance_price_range(t)
            if(upper_range>=data.loc[t,"VWAP"] and lower_range<=data.loc[t,"VWAP"]):
                c1=c1+1
        for t in range(i-1,i+1):
            upper_range,lower_range=resistance_price_range(t)
            if(upper_range>=data.loc[t,"VWAP"] and lower_range<=data.loc[t,"VWAP"]):
                c2=c2+1
        if(c1>=2 and c2>=1):
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

    #bearsh morubozu
    def is_bearish_morubozu(i):
        if(is_bearish(i)):
            body=data["Open"][i]-data["Close"][i]
            up_shadow=data["High"][i]-data["Open"][i]
            low_shadow=data["Close"][i]-data["Low"][i]
            if( up_shadow<=0.01*body and low_shadow<=0.01*body):
                return 1
                
    #EVENING STAR
    def is_evening_star(i):
        def price_range(i):
            return(abs(data["Close"][i]-data["Open"][i]))
        
        pr1=price_range(i-2)
        pr2=price_range(i-1)
        close=data["Close"][i]                                      #closing_of_bearish_session
        half_price=(data["Close"][i-2]+data["Open"][i-2])/2         #halfway_price_of_long_bullish_session
        
        if(is_bullish(i-2) and is_bullish(i-1) and is_bearish(i) and  pr1>=2*pr2 and gap_up(i-1)==1 and gap_down(i)==3 and close<=half_price):
            return 1 
            #strong middle session bullish
        if(is_bullish(i-2) and is_bearish(i-1) and is_bearish(i) and pr1>=2*pr2 and gap_up(i-1)==2 and gap_down(i)==2 and close<=half_price):
            return 1
            #average middle session bearish

    def is_doji1(i):
        if(abs(data["Open"][i]-data["Close"][i])<=(5*abs(data["High"][i]-data["Low"][i]))/100 ):
            return 1
            
    #EVENING DOJI STAR
    def is_evening_doji_star(i):
        def price_range(i):
            return(abs(data["Close"][i]-data["Open"][i]))
        
        pr1=price_range(i-2)
        pr2=price_range(i-1)
        close=data["Close"][i]                                      #closing_of_bearish_session
        half_price=(data["Close"][i-2]+data["Open"][i-2])/2         #halfway_price_of_long_bullish_session
        
        if(is_bullish(i-2) and is_doji1(i-1) and is_bearish(i) and  pr1>=2*pr2 and gap_up(i-1)==1 and gap_down(i)==3 and close<=half_price):
            return 1

    #  the dark cloud cover  pattern
    #{
        #previous candle bullish
        #current candle bearish
        #gapup1 is current candle opening at higher than previous days close
        #gapup2 is current candle opening at higher than previous days high
        #driving condition is that the red candle closes more then halfway up the green candle 
    #}

    def is_dark_cloud_cover(i):
        previous=is_bullish(i-1)
        current=is_bearish(i)
        
        gap_up1=data["Open"][i]>data["Close"][i-1]
        gap_up2=data["Open"][i]>data["High"][i-1]
        def dark_cloud_cover_driving_condition(i):
            open_price_of_previous_bullish_session=data["Open"][i-1]
            halfway_price_of_previous_bullish_session=data["Open"][i-1]+((data["Close"][i-1]-data["Open"][i-1])/2)
            close_price_of_current_bearish_session=data["Close"][i]
            
            if(open_price_of_previous_bullish_session<=close_price_of_current_bearish_session<= halfway_price_of_previous_bullish_session):
                return (1)
            
            
        if(previous and current and dark_cloud_cover_driving_condition(i) and gap_up1): 
            #is_dark_cloud_cover.append(symbol)
            return 1
        if(previous and current and dark_cloud_cover_driving_condition(i) and gap_up2 ): # is_downtrend(i-2)
            #is_dark_cloud_cover.append(symbol)strong
            return 1

    # Bearish harami
    def is_bearish_harami(i):
        def bearish_harami_driving_condition(i):
            if(data["Close"][i-1]>data["Open"][i] and data["Open"][i-1]<data["Close"][i] ):
                return 1
        if( is_bearish(i) and is_bullish(i-1) and bearish_harami_driving_condition(i) ):
            return 1

    #bearish englufing                  
    def is_bearish_engulfing(i):
        if(data["Close"][i-1]<data["Open"][i] and data["Close"][i]<data["Open"][i-1] and is_bearish(i) and is_bullish(i-1)):
            return 1

    def is_strong_hangingman(i):
        i=i-1
        nxt=is_bearish(i-1)
        prev=is_bullish(i+1)
        current=is_bearish(i)
        
        if(current):
            upper_shadow=data["High"][i]-data["Open"][i]
            lower_shadow=data["Close"][i]-data["Low"][i]
            body=abs(data["Close"][i]-data["Open"][i])
            
            if(prev and nxt  and (lower_shadow >2*body) and gap_up(i)==2 and gap_down(i+1)==2 and upper_shadow<0.03*body):
                #strong_hangingman.append(symbol)
                return 1

    #hangingman            
    def is_hangingman(i): 
        i=i-1
        prev=is_bullish(i-1)
        nxt=is_bearish(i+1)
        current=is_bearish(i)
            
        if(current):
            upper_shadow=data["High"][i]-data["Open"][i]
            lower_shadow=data["Close"][i]-data["Low"][i]
            body=abs(data["Close"][i]-data["Open"][i])
            if( prev and nxt and (lower_shadow > 2*body) and upper_shadow< 0.03*body):
                #haningman.append(symbol)
                return 1
                
        if(not current):
            upper_shadow=data["High"][i]-data["Close"][i]
            lower_shadow=data["Open"][i]-data["Low"][i]
            body=abs(data["Open"][i]-data["Close"][i])
            if( prev and nxt and (lower_shadow >2*body)  and upper_shadow< 0.03*body ):
                #hangingman.append(symbol)
                return 1

    def is_weak_hangingman(i):
        current=is_bullish(i)
        if(current):
            upper_shadow=data["High"][i]-data["Close"][i]
            lower_shadow=data["Open"][i]-data["Low"][i]
            body=data["Close"][i]-data["Open"][i]
            if(lower_shadow>=2*body and upper_shadow<0.1*body):
                #weakhangingman.append(symbol)
                return 1
        if(not current):
            upper_shadow=data["High"][i]-data["Open"][i]
            lower_shadow=data["Close"][i]-data["Low"][i]
            body=data["Open"][i]-data["Close"][i]
            if(lower_shadow>=2*body and upper_shadow<0.1*body ):
                #weakhangingman.append(symbol)
                return 1

    #shooting star
    def is_weak_shootingstar(i):
        current=is_bullish(i)
        if(current):
            upper_shadow=data["High"][i]-data["Close"][i]
            lower_shadow=data["Open"][i]-data["Low"][i]
            body=data["Close"][i]-data["Open"][i]
            if(upper_shadow>2*body and lower_shadow<0.05*body):
                #weakshootingstar.append(symbol)
                return 1
        if(not current):
            upper_shadow=data["High"][i]-data["Open"][i]
            lower_shadow=data["Close"][i]-data["Low"][i]
            body=data["Open"][i]-data["Close"][i]
            if(upper_shadow>2*body and lower_shadow<0.05*body ):
                #weakshootingstar.append(symbol)
                return 1

    def is_shootingstar(i): 
        i=i-1
        prev=is_bullish(i-1)
        nxt=is_bearish(i+1)
        current=is_bullish(i)
            
        if(current):
            upper_shadow=data["High"][i]-data["Close"][i]
            lower_shadow=data["Open"][i]-data["Low"][i]
            body=data["Close"][i]-data["Open"][i]
            if( prev and nxt and (upper_shadow > 2*body) and  lower_shadow< 0.03*body):
                #shootingstar.append(symbol)
                return 1
                
        if(not current):
            upper_shadow=data["High"][i]-data["Open"][i]
            lower_shadow=data["Close"][i]-data["Low"][i]
            body=data["Open"][i]-data["Close"][i]
            if( prev and nxt and (upper_shadow >2*body) and  lower_shadow< 0.03*body ):
                #shootingstar.append(symbol)
                return 1
                
    def is_strong_shootingstar(i):
        i=i-1
        prev=is_bearish(i-1)
        nxt=is_bullish(i+1)
        current=is_bearish(i)
        
        if(current):
            upper_shadow=data["High"][i]-data["Open"][i]
            lower_shadow=data["Close"][i]-data["Low"][i]
            body=abs(data["Close"][i]-data["Open"][i])
            t1=gap_up(i)
            t2=gap_down(i+1)
            if( prev and nxt and (upper_shadow>2*body) and t1==2 and t2==2 and (lower_shadow<0.03*body)):
                #shootingstar.append(symbol)
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

    def Inside_Candle(data):
        rows1,columns1=data.shape
        mother_candles=[]
        upper=[]
        lower=[]
        no_of_inside_candles=0
        mother_candle_index=0
        for i in range (1,rows1-1):
            #checking for the patterns 
            check1=(data.loc[i-1,"High"] > data.loc[i,"High"])
            check2=(data.loc[i-1,"Low"] < data.loc[i,"Low"])
            if (check1 and check2 ):
                mother_candles.append(i-1)
                upper.append(data.loc[i-1,"High"])
                lower.append(data.loc[i-1,"Low"])
                no_of_inside_candles=+1
                    
                    
                    
            #checking for the breakouts
            for j in range(0,no_of_inside_candles) :   
                check1=(data.loc[i-1,"Close"] < upper[j]  and data.loc[i,"Close"] >upper[j]) #Upper breakout
                check2=(data.loc[i-1,"Close"] > lower[j]  and data.loc[i,"Close"] <lower[j]) #lower breakout
                mother_candle_index=mother_candles[j]
                if(check2):
                    return 1
                    #print( data.loc[i,"DateTime"] ," broken-out the low of ",data.loc[mother_candle_index,"DateTime"]  )

    
    #creating a output data frame
    stock={"Date":[],"Symbol":[],"Stock_id":[],"Strategy":[]}
    stock=pd.DataFrame(stock)

    Nifty100={
        'Symbol': ['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'IOC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SHREECEM', 'SBIN', 'SUNPHARMA', 'TCS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL', 'ULTRACEMCO', 'WIPRO', 'ACC', 'ABBOTINDIA', 'ADANIGREEN', 'ADANITRANS', 'ALKEM', 'AMBUJACEM', 'AUROPHARMA', 'DMART', 'BAJAJHLDNG', 'BANDHANBNK', 'BANKBARODA', 'BERGEPAINT', 'BIOCON', 'BOSCHLTD', 'CADILAHC', 'COLPAL', 'CONCOR', 'DLF', 'DABUR', 'GICRE', 'GODREJCP', 'HDFCAMC', 'HAVELLS', 'HINDPETRO', 'HINDZINC', 'ICICIGI', 'ICICIPRULI', 'IGL', 'INDUSTOWER', 'NAUKRI', 'INDIGO', 'LTI', 'LUPIN', 'MARICO', 'MOTHERSUMI', 'MUTHOOTFIN', 'NMDC', 'OFSS', 'PETRONET', 'PIDILITIND', 'PEL', 'PFC', 'PGHH', 'PNB', 'SBICARD', 'SIEMENS', 'TATACONSUM', 'TORNTPHARM', 'UBL', 'MCDOWELL-N'], 
    
        'Id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 
        96, 97, 98, 99, 100]
    }
                

    for i in range(0,100):
        
        """
        data = td_app.get_historic_data( i, duration='3 M', bar_size='15 min')
        data=pd.DataFrame(data)
        data.columns=["Time","Open","High","Low","Close","Volume","oi"]
        data=data[["Time","Open","High","Low","Close","Volume"]]


        data.set_index("Time",inplace=True)
        data=data.resample("1D").agg({"Open":"first","High":max,"Low":min,"Close":"last","Volume":sum})
        data.dropna(inplace=True)
        data.reset_index(inplace=True)
        rows,columns=data.shape
        """

        today = date.today()
        today = today.strftime("%Y,%m,%d")
        

        data = get_history(symbol=Nifty100["Symbol"][i], start=date(2020,1,1), end=date(2021,4,6))
        print(data)
        data.reset_index(inplace=True)
        data=Calculation_Of_Techincal_Indicators(data)
        rows,column=data.shape
        data=Calculation_Of_Techincal_Indicators(data)

        """
        for t in range(len(gan_static_levels)):
            if(data.loc[rows-1,"Close"]>=gan_static_levels[t] and data.loc[rows-1,"Close"]<=gan_static_levels[t+1]):
                stop_loss=gan_static_levels[t+1]
                entry=gan_static_levels[t]
                target1=gan_static_levels[t-1]
                target2=gan_static_levels[t-2]
                target3=gan_static_levels[t-3]
                break
        """

        if(sma_5_20_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-5-20 crossover"},ignore_index=True)
        if(sma_10_30_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-10-30 crossover"},ignore_index=True)
        if(sma_20_50_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-20-50 crossover"},ignore_index=True)
        if(ema_5_20_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-5-20 crossover"},ignore_index=True)
        if(ema_20_50_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-20-50 crossover"},ignore_index=True)
        if(ema_10_30_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-10-30 crossover"},ignore_index=True)
        if(sma_20_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-20 pricecrossover"},ignore_index=True)
        if(sma_50_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-50 pricecrossover"},ignore_index=True)
        if(ema_20_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-20 pricecrossover"},ignore_index=True)
        if(ema_50_price_crossover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-50 pricecrossover"},ignore_index=True)
    
        if(sma_20_resistance(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SMA-20 resistance"},ignore_index=True)
        if(ema_20_resistance(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"EMA-20 resistance"},ignore_index=True)
        
        if(MACD_signal(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"MACD"},ignore_index=True)
        if(Bollinger_Band_signal(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"BollingerBand" },ignore_index=True)     
        if(rsi_overbought(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"RSI-overbought" },ignore_index=True)
        if(triple_supertrend(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"triple supertrend change-14-10-7"},ignore_index=True)
        if(supertrend_change(rows-1)==14):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SuperTrend change-14"},ignore_index=True)
        if(supertrend_change(rows-1)==10):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"SuperTrend change-10"},ignore_index=True)
        if(is_dark_cloud_cover(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"DarkCloudCover"},ignore_index=True)
        if(is_bearish_engulfing(rows-1) ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Bearish-Engulfing"},ignore_index=True)
        if(is_bearish_harami(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Bearish-Harami"},ignore_index=True)
        if(is_bearish_morubozu(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Bearish-morubozu"},ignore_index=True)
        if(is_hangingman(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"HangingMan"},ignore_index=True)
        if(is_weak_hangingman(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Weak-Hangingman"},ignore_index=True)
        if(is_strong_hangingman(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Strong-Hangingman"},ignore_index=True)    
        if(is_shootingstar(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"ShootingStar"},ignore_index=True)
        if(is_strong_shootingstar(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Strong-Shootingstar"},ignore_index=True)
        if(is_weak_shootingstar(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Weak-Shootingstar"},ignore_index=True)
        if(is_evening_doji_star(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Evening-Doji-star"},ignore_index=True)
        if(is_evening_star(rows-1)):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Evening-Star"},ignore_index=True)
        if(doji(rows-1)==1 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Dragonfly-Doji"},ignore_index=True)
        if(doji(rows-1)==2 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Graveyard-Doji"},ignore_index=True)
        if(doji(rows-1)==3 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Doji"},ignore_index=True)
        if(doji(rows-1)==7 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Price-Doji"},ignore_index=True)
        if(doji(rows-1)==8 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Perfect-Doji"},ignore_index=True)
        if(doji(rows-1)==9 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Perfect-Dragonfly-Doji"},ignore_index=True)
        if(doji(rows-1)==10 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Perfect-Graveyard-Doji"},ignore_index=True)
        if(doji(rows-1)==11 ):
            stock=stock.append({"Date":data.loc[rows-1,"Date"],"Symbol":Nifty100["Symbol"][i],"Stock_id":Nifty100["Id"][i],"Strategy":"Perfect-Common-Doji"},ignore_index=True)
    
    return(stock)

def populate_databases():
    

    
    

    mydb=mysql.connector.connect(host='localhost',user='root',password='Yash@mysql99',auth_plugin='mysql_native_password')
    cur=mydb.cursor()

    q1="USE  finixsanlabs   "
    cur.execute(q1)
    
    
    
    stock_buy=EOD_buy()
    
    
    q2="INSERT INTO eod_bullishstocks(date,strategy,stock_id_id) VALUES (%s,%s,%s) "

  

    b1=[]
    for i in range(len(stock_buy)):
        
        b1.append((stock_buy.loc[i,"Date"],stock_buy.loc[i,"Strategy"],int(stock_buy.loc[i,"Stock_id"])))
    
    cur.executemany(q2,b1)
    mydb.commit()

    """
    stock_sell=sell_EOD()
   
    
    
    q3="INSERT INTO eod_bearishstocks(date,strategy,stock_id_id) VALUES (%s,%s,%s) "

    cur.execute(q1)

    b2=[]
    for i in range(len(stock_sell)):
        
        b2.append((stock_sell.loc[i,"Date"],stock_sell.loc[i,"Strategy"],int(stock_sell.loc[i,"Stock_id"])))
    
    cur.executemany(q3,b2)
    mydb.commit()
    """
    
populate_databases()