import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data=pd.read_csv("test.csv")
rows,columns=data.shape


def is_bearish(i):
    return(data["Open"][i]>data["Close"][i])
    
def is_bullish(i):
    return(data["Open"][i]<data["Close"][i])

def is_open_eqaul_close(i):
    return(data["Open"][i]==data["Close"][i])

def doji(i):
    body=abs(data["Open"][i]-data["Close"][i])
    total_candle=data["High"][i]-data["Low"][i]
    if(body<0.1*total_candle):
        if(is_bullish(i)):
            up_shadow=data["High"][i]-data["Close"][i]
            low_shadow=data["Open"][i]-data["Low"][i]
        
            if(up_shadow<0.1*low_shadow):
                #dragonfly_doji.append()
                print(data["Date"][i],"dragonfly doji")
                #return 1
            elif(low_shadow<0.1*up_shadow):
                #graveyard_doji.append()
                print(data["Date"][i],"graveyard  doji")
                #return 1
            else:
                #common_doji.append()
                print(data["Date"][i],"common_doji ")
                #return 1
        
        if(is_bearish(i)):
            up_shadow=data["High"][i]-data["Open"][i]
            low_shadow=data["Close"][i]-data["Low"][i]
               
            if(up_shadow<0.1*low_shadow):
                #dragonfly_doji.append()
                print(data["Date"][i],"dragonfly doji")
                #return 1
            elif(low_shadow<0.1*up_shadow):
                #graveyard_doji.append()
                print(data["Date"][i],"graveyard  doji")
                #return 1
            else:
                #common_doji.append()
                print(data["Date"][i],"common_doji ")
                #return 1 
                
    if(is_open_eqaul_close(i)):
        up_shadow=data["High"][i]-data["Close"][i]
        low_shadow=data["Open"][i]-data["Low"][i]
        if(data["High"][i]==data["Low"][i]):
            #price_doji.append()
            print(data["Date"][i],"price doji")
            #return 1
        elif(up_shadow==low_shadow):
            #perfect_doji.append()
            print(data["Date"][i],"perfect doji")
            #return 1
        elif(up_shadow<0.1*low_shadow):
                #dragonfly_doji.append()
                print(data["Date"][i],"dragonfly doji")
                #return 1
        elif(low_shadow<0.1*up_shadow):
                #graveyard_doji.append()
                print(data["Date"][i],"graveyard  doji")
                #return 1
        else:
            #common_doji.append()
            print(data["Date"][i],"common_doji ")
            #return 1 
    
