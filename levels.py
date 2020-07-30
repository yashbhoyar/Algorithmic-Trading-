import pandas as pd 
import csv 
import time
import os
import numpy
import math

x=str(input("Enter file name: "))
#z=str(input("Enter volatility file name:"))
y=str(input("Enter output file name:"))
x=x+".csv"
#z=z+".csv"
y=y+".csv"
 

#main file ohlc 
data=pd.read_csv(x)
data=pd.DataFrame(data)

###---->>>> this is to be used when calculating data from nse website for nifty50 data and banknifty
def set_data1(data):
    data.rename(columns={"Last Traded Price":"Close"},inplace=True)
    data=data[["Symbol","Open","High","Low","Close",]] 
    #removing commas from the data and typecasting it to float 
    data["Open"]=data["Open"].str.replace(",","").astype(float)
    data["High"]=data["High"].str.replace(",","").astype(float)
    data["Low"]=data["Low"].str.replace(",","").astype(float)
    data["Close"]=data["Close"].str.replace(",","").astype(float)
    return (data)

def set_data2(data):
    data=data[["Date","Open","High","Low","Close",]]
    return (data)
    ##----->>>>>this is used for yahoo finace data for calculating levels for individual stocks

#volatility file
#vol=pd.read_csv(z)
#vol=pd.DataFrame(vol)
#vol.columns=df.columns.str.replace(" ","_")
#vol.rename(columns={"_Applicable_Annualised_Volatility_(N)_=_Max_(F_or_L)":"Volatility"},inplace=True)

#def fib_levels(data,vol):
    #Range=pd.Series(vol["Volatility"]*(math.sqrt(1/365)*data["Close"])).astype("float")


   # Fib_0.236=pd.Series(data["Close"]+Range*0.236)

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

def gan_degrees(data):
    gn45 = pd.Series(data['Close'].map(lambda z: (math.sqrt(z)-45/180)**2))
    gn90=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-90/180)**2))
    gn135=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-135/180)**2))
    gn180=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-180/180)**2))
    gn225=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-225/180)**2))
    gn270=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-270/180)**2))
    gn315=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-315/180)**2))
    gn360=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-360/180)**2))
    gn405=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-405/180)**2))
    gn450=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-450/180)**2))
    gn495=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-495/180)**2) )
    gn540=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-540/180)**2)) 
    gn585=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-585/180)**2)) 
    gn630=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-630/180)**2))
    gn675=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-675/180)**2) )
    gn720=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)-720/180)**2))
    g45=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+45/180)**2))
    g90=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+90/180)**2))
    g135=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+135/180)**2))
    g180=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+180/180)**2))
    g225=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+225/180)**2))
    g270=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+270/180)**2))
    g315=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+315/180)**2))
    g360=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+360/180)**2))
    g450=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+450/180)**2))
    g405=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+405/180)**2))
    g495=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+495/180)**2) )
    g540=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+540/180)**2) )
    g585=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+585/180)**2) )
    g630=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+630/180)**2))
    g675=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+675/180)**2) )
    g720=pd.Series(data['Close'].map(lambda z:(math.sqrt(z)+720/180)**2) )
    gan={"gn720":gn720,"gn675":gn675,"gn630":gn630,"gn585":gn585,"gn540":gn540,"gn495":gn495,"gn405":gn405,"gn360":gn360,"gn315":gn315,"gn270":gn270,"gn225":gn225,"gn180":gn180,"gn135":gn135,"gn90":gn90,"gn45":gn45,"g45":g45,"g90":g90,"g135":g135,"g180":g180,"g225":g225,"g270":g270,"g315":g315,"g360":g360,"g405":g405,"g450":g450,"g495":g495,"g540":g540,"g585":g585,"g630":g630,"g675":g675,"g720":g720}
    gan=pd.DataFrame(gan)
    data= data.join(gan)
    return(data)



def tcp_bcp(data):
    FPP = pd.Series((data['High'] + data['Low'] + data['Close']) / 3)
    bcp=pd.Series((data["High"]+data["Low"])/2)
    tcp=pd.Series((2*FPP)-bcp)
    tcp_bcp={"bcp":bcp,"tcp":tcp}
    tcp_bcp=pd.DataFrame(tcp_bcp)
    data=data.join(tcp_bcp)
    return data

#setting data
x=int(input("Enter 1.If data taken from nse site \n 2.If data taken from yahoo finance :"))
if(x==1):
    data=set_data1(data)
if(x==2):
    data=set_data2(data)

#applying levels
data=PPSR(data)
data=FIB_PPSR(data)
data=tcp_bcp(data)
data=gan_degrees(data)

#setting columns in order
columns=["Symbol","Date","Open","High","Close","PP","R1","R2","R3","S1","S2","S3","FPP","FR1","FR2","FR3","FS1","FS2","FS3","bcp","tcp","gn720","gn675","gn630","gn585","gn540","gn495","gn405","gn360","gn315","gn270","gn225","gn180","gn135","gn90","gn45","g45","g90","g135","g180","g225","g270","g315","g360","g405","g450","g495","g540","g585","g630","g675","g720"]
data=data.reindex(columns=columns)

#saving file
data.to_csv(y)
