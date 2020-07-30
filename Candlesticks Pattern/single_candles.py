import pandas as pd 
import csv 
import time
import os
import numpy
data=pd.read_csv("test.csv")
data=pd.DataFrame(data)
data
rows,columns=data.shape


def is_bearish(i):
    return(data["Open"][i]>data["Close"][i])
    
def is_bullish(i):
    return(data["Open"][i]<data["Close"][i])

def is_open_eqaul_close(i):
    return(data["Open"][i]==data["Close"][i])


rows,columns=data.shape
bullish_engulfing=[]


#shooting star
def is_weak_shootingstar(i):
    current=is_bullish(i)
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if(upper_shadow>2*body and lower_shadow<0.05*body):
            #weakshootingstar.append(symbol)
            print(data["Date"][i])
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if(upper_shadow>2*body and lower_shadow<0.05*body ):
            #weakshootingstar.append(symbol)
            print(data["Date"][i])

def is_shootingstar(i): 
    prev=is_bullish(i-1)
    nxt=is_bearish(i+1)
    current=is_bullish(i)
        
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if( prev and nxt and (upper_shadow > 2*body) and  lower_shadow< 0.03*body):
            #shootingstar.append(symbol)
            print(data["Date"][i])
            
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if( prev and nxt and (upper_shadow >2*body) and  lower_shadow< 0.03*body ):
            #shootingstar.append(symbol)
            print(data["Date"][i])
            
def is_shootingstar(i):
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
            print(data["Date"][i])

#hanging man
def is_strong_hangingman(i):
    nxt=is_bearish(i-1)
    prev=is_bullish(i+1)
    current=is_bearish(i)
    
    if(current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=abs(data["Close"][i]-data["Open"][i])
        
        if(prev and nxt  and (lower_shadow >2*body) and gap_up(i)==2 and gap_down(i+1)==2 and upper_shadow<0.03*body):
            #strong_hangingman.append(symbol)
            print(data["Date"][i])

            
def is_hangingman(i): 
    prev=is_bullish(i-1)
    nxt=is_bearish(i+1)
    current=is_bearish(i)
        
    if(current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=abs(data["Close"][i]-data["Open"][i])
        if( prev and nxt and (lower_shadow > 2*body) and upper_shadow< 0.03*body):
            #haningman.append(symbol)
            print(data["Date"][i])
            
    if(not current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=abs(data["Open"][i]-data["Close"][i])
        if( prev and nxt and (lower_shadow >2*body)  and upper_shadow< 0.03*body ):
            #hangingman.append(symbol)
            print(data["Date"][i])

def is_weak_hangingman(i):
    current=is_bullish(i)
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if(lower_shadow>=2*body and upper_shadow<0.1*body):
            #weakhangingman.append(symbol)
            print(data["Date"][i])
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if(lower_shadow>=2*body and upper_shadow<0.1*body ):
            #weakhangingman.append(symbol)
            print(data["Date"][i])

#Inverted hammer
def is_weak_inverted_hammer(i):
    current=is_bullish(i)
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if(upper_shadow>2*body and lower_shadow<0.05*body):
            #weak_inverted_hammer.append(symbol)
            print(data["Date"][i])
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if(upper_shadow>2*body and lower_shadow<0.05*body ):
            #weakhammer.append(symbol)
            print(data["Date"][i])

def is_inverted_hammer(i): 
    prev=is_bearish(i-1)
    nxt=is_bullish(i+1)
    current=is_bullish(i)
        
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if( prev and nxt and (upper_shadow > 2*body) and  lower_shadow< 0.05*body):
            #hammer.append(symbol)
            print(data["Date"][i])
            
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if( prev and nxt and (upper_shadow >2*body) and  lower_shadow< 0.05*body ):
            #hammer.append(symbol)
            print(data["Date"][i])
            
def is_strong_inverted_hammer(i):
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
            #strong_hammer.append(symbol)
            print(data["Date"][i])


#HAMMER candlestick patterns
def is_weak_hammer(i):
    current=is_bullish(i)
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if(lower_shadow>=2*body and upper_shadow<0.1):
            #weakhammer.append(symbol)
            print(data["Date"][i])
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if(lower_shadow>=2*body and upper_shadow<0.1 ):
            #weakhammer.append(symbol)
            print(data["Date"][i])

def is_hammer(i): 
    prev=is_bearish(i-1)
    nxt=is_bullish(i+1)
    current=is_bullish(i)
        
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        if( prev and nxt and (lower_shadow > 2*body) and data["Low"][i-1]<data["Close"][i]<data["Close"][i-1] and data["Low"][i-1]<data["Low"][i] and upper_shadow< 0.05*body):
            #hammer.append(symbol)
            print(data["Date"][i])
            
    if(not current):
        upper_shadow=data["High"][i]-data["Open"][i]
        lower_shadow=data["Close"][i]-data["Low"][i]
        body=data["Open"][i]-data["Close"][i]
        if( prev and nxt and (lower_shadow >2*body) and data["Low"][i-1]<data["Open"][i]<data["Close"][i-1] and data["Low"][i-1]<data["Low"][i] and upper_shadow< 0.05*body ):
            #hammer.append(symbol)
            print(data["Date"][i])
            
def is_strong_hammer(i):
    prev=is_bearish(i-1)
    nxt=is_bullish(i+1)
    current=is_bullish(i)
    
    if(current):
        upper_shadow=data["High"][i]-data["Close"][i]
        lower_shadow=data["Open"][i]-data["Low"][i]
        body=data["Close"][i]-data["Open"][i]
        
        if(prev and nxt  and (lower_shadow >2*body) and gap_up(i+1)==1 and gap_down(i)==1 and upper_shadow<=0.01*body):
            #strong_hammer.append(symbol)
            print(data["Date"][i])
        
#marubozu

def is_bearish(i):
    return(data["Open"][i]>data["Close"][i])
    
def is_bullish(i):
    return(data["Open"][i]<data["Close"][i])

bullish_morubozu=[]
bearish_morubozu=[]

def is_bullish_morubozu(i):
    if(is_bullish(i)):
        body=data["Close"][i]-data["Open"][i]
        up_shadow=data["High"][i]-data["Close"][i]
        low_shadow=data["Open"][i]-data["Low"][i]
        if( up_shadow<=0.01*body and low_shadow<=0.01*body):
            #bullish_morubozu.append()
            #return 1
            #print(data["Date"][i],"Bullish")

def is_bearish_morubozu(i):
    if(is_bearish(i)):
        body=data["Open"][i]-data["Close"][i]
        up_shadow=data["High"][i]-data["Open"][i]
        low_shadow=data["Close"][i]-data["Low"][i]
        if( up_shadow<=0.01*body and low_shadow<=0.01*body):
            #bearish_morubozu.append()
            #return 1
            #print(data["Date"][i],"Bearish")

    
        