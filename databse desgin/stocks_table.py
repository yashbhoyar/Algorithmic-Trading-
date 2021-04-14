import pandas as pd
import mysql.connector



#creating a connection:
mydb=mysql.connector.connect(host='localhost',user='root',password='Yash@mysql99',auth_plugin='mysql_native_password')
cur=mydb.cursor()

#taking the database into consideration
q1="USE finixsanlabs"
cur.execute(q1)

#exporting all the company names and symbol in stocks table:
data=pd.read_csv("Stocks.csv")

q2="INSERT INTO eod_stocks (id,symbol,companyName,broadIndex,sectoralIndex,industry) VALUES(%s,%s,%s,%s,%s,%s)"


b1=[]
for i in range(len(data)):
    b1.append((i+1,data.loc[i,"Symbol"],data.loc[i,"Company_Name"],data.loc[i,"Broad_Index"],data.loc[i,"Sectoral_Index"],data.loc[i,"Industry"]))
print(b1)
cur.executemany(q2,b1)
mydb.commit()