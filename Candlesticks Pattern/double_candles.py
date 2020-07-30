###Two candles pattern
#   1. the peircing pattern
#    2. bullish harami
#    3. bearish harami
#    4. bullish engulfing
#    5. bearish engulfing
#   6.dark cloud cover

import pandas as pd 
import csv 
import time
import os

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

#Two candles pattern:

#  1.the peircing pattern
#{
    #previous candle bearish
    #current candle bullish
    #gapdown1 is current candle opening at lower than previous days close
    #gapdown2 is current candle oprning at lower than previous days low
    #driving condition is that the green candle closes more then halfway up the red candle 
#}

def is_peircing(i):
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
        
        
    if(previous and current and peircing_driving_condition(i) and gap_down1): #and is_downtrend(i-2)
        #is_peircing.append(symbol)
        print(data["Date"][i])
        #is_downtrend(i-1)
    elif(previous and current and peircing_driving_condition(i) and gap_down2 ): # is_downtrend(i-2)
        #is_peircing.append(symbol)strong
        print(data["Date"][i])
        #is_downtrend(i-1)
    
for i in range (1,rows):
    is_peircing(i)        
        
#  1.the dark cloud cover  pattern
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
        
        
    if(previous and current and dark_cloud_cover_driving_condition(i) and gap_up1): #and is_downtrend(i-2)
        #is_dark_cloud_cover.append(symbol)
        print(data["Date"][i])
        #is_downtrend(i-1)
    elif(previous and current and dark_cloud_cover_driving_condition(i) and gap_up2 ): # is_downtrend(i-2)
        #is_dark_cloud_cover.append(symbol)strong
        print(data["Date"][i])
        #is_downtrend(i-1)
    
for i in range (1,rows):
    is_peircing(i)        
        
# 2. Bearish harami
def is_bearish_harami(i):
    def bearish_harami_driving_condition(i):
        if(data["Close"][i-1]>data["Open"][i] and data["Open"][i-1]<data["Close"][i] ):
            return 1
    if( is_bearish(i) and is_bullish(i-1) and bearish_harami_driving_condition(i) ):
        print(data["Date"][i])

# 3. Bullish harami
def is_bullish_harami(i):
    def bullish_harami_driving_condition(i):
        if(data["Close"][i-1]<data["Open"][i] and data["Open"][i-1]>data["Close"][i] ):
            return 1
    if( is_bearish(i-1) and is_bullish(i) and bullish_harami_driving_condition(i) ):
        print(data["Date"][i])        

#4.Bullish engulfing
def is_bullish_engulfing(i):
    if(data["Close"][i-1]>data["Open"][i] and data["Close"][i]>data["Open"][i-1] and is_bearish(i-1) and is_bullish(i)):
        print(data["Date"][i])
        


#5.bearish englufing                  
def is_bearish_engulfing(i):
    if(data["Close"][i-1]<data["Open"][i] and data["Close"][i]<data["Open"][i-1] and is_bearish(i) and is_bullish(i-1)):
        print(data["Date"][i])

print("peircing")
for i in range (1,rows):
    is_peircing(i) 

print("is_dark_cloud_cover")
for i in range (1,rows):
    is_dark_cloud_cover(i) 
print("bearish harami")
for i in range (1,rows):
    is_bearish_harami(i)    
    
print("bullish harami")    
for i in range (1,rows):
    is_bullish_harami(i)
    
print("bullish engulfing")    
for i in range (1,rows):
    is_bullish_engulfing(i)
    
print("bearish engulfing")    
for i in range (1,rows):
    is_bearish_engulfing(i)    