import pandas as pd
from datetime import datetime,date
from datetime import timedelta
import numpy as np
from findiff import FinDiff 
from nsepy import get_history
import json 

#dates
to_date=datetime.now()
to_date=datetime.strftime(to_date,'%Y,%m,%d')
to_date=datetime.strptime(to_date,'%Y,%m,%d')
from_date=to_date - timedelta(days=60)

#Moving average
def MA(data,n):
    x="MA"+str(n)
    data[x]=data["Close"].rolling(n).mean()
    return data

#Exponential Moving average
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
    data[x]=data["Volume"].rolling(n).mean()
    return data

#EMA for supertrend 
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

#ATR
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

#supertrend
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

def PPSR(data):  
    PP = pd.Series((data['High'] + data['Low'] + data['Close']) / 3)  
    R1 = pd.Series(2 * PP - data['Low'])  
    S1 = pd.Series(2 * PP - data['High'])  
    R2 = pd.Series(PP + data['High'] - data['Low'])  
    S2 = pd.Series(PP - data['High'] + data['Low'])  
    R3 = pd.Series(data['High'] + 2 * (PP - data['Low']))  
    S3 = pd.Series(data['Low'] - 2 * (data['High'] - PP))  
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)  
    data= data.join(PSR)  
    return data

def FIB_PPSR(data):  
    FPP = pd.Series((data['High'] + data['Low'] + data['Close']) / 3)  
    FR1 = pd.Series( FPP + ((data["High"]-data['Low'])*0.382))  
    FS1 = pd.Series( FPP - ((data["High"]-data['Low'])*0.382))  
    FR2 = pd.Series( FPP + ((data["High"]-data['Low'])*0.618))  
    FS2 = pd.Series( FPP - ((data["High"]-data['Low'])*0.618))  
    FR3 = pd.Series( FPP + ((data["High"]-data['Low'])*1))  
    FS3 = pd.Series( FPP - ((data["High"]-data['Low'])*1))  
    fpsr = {'FPP':FPP, 'FR1':FR1, 'FS1':FS1, 'FR2':FR2, 'FS2':FS2, 'FR3':FR3, 'FS3':FS3}  
    FPSR = pd.DataFrame(fpsr)  
    data= data.join(FPSR)  
    return data

def stochastic(data):
    data['14-high'] = data['High'].rolling(14).max()
    data['14-low'] = data['Low'].rolling(14).min()
    data['%K'] = (data['Close'] - data['14-low'])*100/(data['14-high'] - data['14-low'])
    data['%D'] = data['%K'].rolling(3).mean()
    return(data)

def Calculation_Of_Techincal_Indicators(data):
    
    #sma
    data=MA(data,5)
    data=MA(data,9)
    data=MA(data,10)
    data=MA(data,15)
    data=MA(data,20)
    data=MA(data,21)
    data=MA(data,30)
    data=MA(data,50)
    data=MA(data,55)
    data=MA(data,60)
    data=MA(data,100)
    data=MA(data,120)
    data=MA(data,150)
    data=MA(data,180)
    data=MA(data,200)

    #ema
    data=EMA(data,5)
    data=EMA(data,9)
    data=EMA(data,10)
    data=EMA(data,15)
    data=EMA(data,20)
    data=EMA(data,21)
    data=EMA(data,30)
    data=EMA(data,50)
    data=EMA(data,55)
    data=EMA(data,60)
    data=EMA(data,100)
    data=EMA(data,120)
    data=EMA(data,150)
    data=EMA(data,180)
    data=EMA(data,200)

    #calculating rsi
    data=RSI(data,14)


    #calculating bollinger_bands
    data=Bolinger_Bands(data,20,2)

    #calculating MACD
    data=MACD(data,12,26,9)

    #calculating_vma
    data=VMA(data,10)
    data=VMA(data,20)

    #supertrend
    data=SuperTrend(data, 14, 3, ohlc=['Open', 'High', 'Low', 'Close'])
    data=SuperTrend(data, 10, 3, ohlc=['Open', 'High', 'Low', 'Close'])
    data=SuperTrend(data, 7, 3, ohlc=['Open', 'High', 'Low', 'Close'])

    data=PPSR(data)
    data=FIB_PPSR(data)

    data=stochastic(data)

    return data

Nifty50=['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BRITANNIA',
                'CIPLA', 'COALINDIA', 'DRREDDY', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HEROMOTOCO', 
                'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY',  'KOTAKBANK',
                'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'RELIANCE', 'SBIN', 'SUNPHARMA',
                'TCS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL',  'WIPRO']

data=get_history(symbol="SBIN",start=from_date,end=to_date)

data.reset_index(inplace=True)

stock=Calculation_Of_Techincal_Indicators(data)

stock.to_csv("sample2.csv")