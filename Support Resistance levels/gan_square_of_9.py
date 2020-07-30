import pandas as pd
import math 
pd.set_option('display.max_column',12)
ltp=float(input("Enter the current market price:"))
middle_price=lambda x: (int(math.sqrt(ltp))-1)
middle_price=middle_price(ltp)


data={"middle_price":[middle_price**2]}
data=pd.DataFrame(data)

prices=[]
for i in range (45,1125,45):
    prices.append((middle_price+i/360)**2)


for t in range(0,len(prices)):
    if(ltp>=prices[t] and ltp<prices[t+1]):
        lower_than_ltp=prices[t]
        higher_than_ltp=prices[t+1]
        
        R1=pd.Series(prices[t+2])
        R2=pd.Series(prices[t+3])
        R3=pd.Series(prices[t+4])
        R4=pd.Series(prices[t+5])
        R5=pd.Series(prices[t+6])
        S1=pd.Series(prices[t-1])
        S2=pd.Series(prices[t-2])
        S3=pd.Series(prices[t-3])
        S4=pd.Series(prices[t-4])
        S5=pd.Series(prices[t-5])
 
        new_data={"R1":R1,"R2":R2,"R3":R3,"R4":R4,"R5":R5,"S1":S1,"S2":S2,"S3":S3,"S4":S4,"S5":S5}
        new_data=pd.DataFrame(new_data)
        data=data.join(new_data)
        break
    
print(data)
print("recommendations:Buy")
print("Entry:At/Above:",higher_than_ltp)
print("StopLoss:",lower_than_ltp)

print("recommendations:sell")
print("Entry:At/Below:",lower_than_ltp)
print("Stoploss:",higher_than_ltp)

input("press enter to exit")