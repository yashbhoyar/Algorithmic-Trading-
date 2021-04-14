import pandas as pd

data=pd.read_csv(r"C:\Users\Yash\Desktop\database design\Stocks.csv")

stocks_symbol_with_id={"Symbol":[],"Id":[]}

for i in range(len(data)):
    
    stocks_symbol_with_id["Symbol"].append(data.loc[i,"Symbol"])
    stocks_symbol_with_id["Id"].append(i+1)

    
print(stocks_symbol_with_id)
