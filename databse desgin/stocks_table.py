import pandas as pd
import mysql.connector
import numpy as np


#creating a connection:
mydb=mysql.connector.connect(host='localhost',user='root',password='Yash@mysql99',auth_plugin='mysql_native_password')
cur=mydb.cursor()


#exporting all the company names and symbol in stocks table:
data=pd.read_excel("Stocks.xlsx",engine='openpyxl')
q1="USE finixsanlabs"
q2="INSERT INTO eod_analysis_stocks (symbol,company,) VALUES(%s,%s,%s)"
cur.execute(q7)

b1=[]
for i in range(len(data)):
    b1.append((i+1,data.loc[i,"Symbol"],data.loc[i,"Company Name"]))
   
cur.executemany(q8,b1)
mydb.commit()
