import pandas as pd 
import csv 
import time
import os

data=pd.read_csv("test.csv")
data=pd.DataFrame(data)

rows,columns=data.shape


def is_bearish(i):
    return(data["Open"][i]>data["Close"][i])
    
def is_bullish(i):
    return(data["Open"][i]<data["Close"][i])

def is_open_eqaul_close(i):
    return(data["Open"][i]==data["Close"][i])

def gap_down(i):                                               #gap_down
    
    if(is_bearish(i-1) and is_bullish(i)):               #  1 bearish bulish
        if(data["Close"][i-1]>data["Close"][i]):          #  2 bearish bearish
            return (1)                                   #  3 bullish bearish
    if(is_bearish(i-1) and is_bearish(i)):               
        if(data["Close"][i-1]>data["Open"][i]):
            return (2)
    if (is_bullish(i-1) and is_bearish(i)):
        if(data["Open"][i-1]>data["Open"][i]):
            return (3)
         
            
def gap_up(i):                                               #gap_up
                                                        #  1 bulish bullish
    if(is_bullish(i-1) and is_bullish(i)):              #  2 bullish bearish
        if(data["Close"][i-1] <data["Open"][i]):        #  3 bearish bullish
            return (1)
    if(is_bullish(i-1) and is_bearish(i)):
        if(data["Close"][i]>data["Close"][i-1]):
            return(2)
    if(is_bearish(i-1) and is_bullish(i)):
        if(data["Open"][i-1]<data["Open"][i]):
            return(3)

#DOJI
def is_doji(i):
    if(abs(data["Open"][i]-data["Close"][i])<=(5*abs(data["High"][i]-data["Low"][i]))/100 ):
        print(data["Date"][i])

def is_doji1(i):
    if(abs(data["Open"][i]-data["Close"][i])<=(5*abs(data["High"][i]-data["Low"][i]))/100 ):
        return 1

#MORNING STAR 
def is_morning_star(i):
    def price_range(i):
        return(abs(data["Close"][i]-data["Open"][i]))
    
    pr1=price_range(i-2)
    pr2=price_range(i-1)
    close=data["Close"][i]                                  #closing of bullish session
    half_price=(data["Close"][i-2]+data["Open"][i-2])/2     #half_price_of_long_bearish_session
    
    if(is_bearish(i-2) and is_bullish(i-1) and is_bullish(i) and pr1>=2*pr2 and gap_down(i-1)==1 and gap_up(i)==1 and close>=half_price):
        print(data["Date"][i])
        #strong_middle_candle_bullish
    if(is_bearish(i-2) and is_bearish(i-1) and is_bullish(i) and pr1>=2*pr2 and gap_down(i-1)==2 and gap_up(i)==3 and close>=half_price):
        print(data["Date"][i])    
        #average_middle_candle_bearish

#morning doji star
def is_morning_doji_star(i):
    def price_range(i):
        return(abs(data["Close"][i]-data["Open"][i]))
    
    pr1=price_range(i-2)
    pr2=price_range(i-1)
    close=data["Close"][i]                                  #closing of bullish session
    half_price=(data["Close"][i-2]+data["Open"][i-2])/2     #half_price_of_long_bearish_session
    
    if(is_bearish(i-2) and is_doji1(i-1) and is_bullish(i) and pr1>=2*pr2 and gap_down(i-1)==1 and gap_up(i)==1 and close>=half_price):
        print(data["Date"][i])


def is_evening_star(i):
    def price_range(i):
        return(abs(data["Close"][i]-data["Open"][i]))
    
    pr1=price_range(i-2)
    pr2=price_range(i-1)
    close=data["Close"][i]                                      #closing_of_bearish_session
    half_price=(data["Close"][i-2]+data["Open"][i-2])/2         #halfway_price_of_long_bullish_session
    
    if(is_bullish(i-2) and is_bullish(i-1) and is_bearish(i) and  pr1>=2*pr2 and gap_up(i-1)==1 and gap_down(i)==3 and close<=half_price):
        print(data["Date"][i]) 
        #strong middle session bullish
    if(is_bullish(i-2) and is_bearish(i-1) and is_bearish(i) and pr1>=2*pr2 and gap_up(i-1)==2 and gap_down(i)==2 and close<=half_price):
        print(data["Date"][i])
        #average middle session bearish

#evening doji star
def is_evening_doji_star(i):
    def price_range(i):
        return(abs(data["Close"][i]-data["Open"][i]))
    
    pr1=price_range(i-2)
    pr2=price_range(i-1)
    close=data["Close"][i]                                      #closing_of_bearish_session
    half_price=(data["Close"][i-2]+data["Open"][i-2])/2         #halfway_price_of_long_bullish_session
    
    if(is_bullish(i-2) and is_doji1(i-1) and is_bearish(i) and  pr1>=2*pr2 and gap_up(i-1)==1 and gap_down(i)==3 and close<=half_price):
        print(data["Date"][i])

print("evening star")
for i in range(2,rows):
    is_evening_star(i)
    
print("morning star")    
for i in range(2,rows):
    is_morning_star(i)
    
print("Doji")    
for i in range(0,rows):
    is_doji(i)

print("morning doji star")     
for i in range(2,rows):
    is_morning_doji_star(i)

print("evening  doji star")
for i in range (2,rows):
    is_evening_doji_star(i)