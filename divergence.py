import pandas as pd
from datetime import datetime,date
from datetime import timedelta
import numpy as np
from findiff import FinDiff 
from truedata_ws.websocket.TD import TD

td_app = TD('FYERS452', 'jxVM5P54')

Nifty50=['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BRITANNIA',
            'CIPLA', 'COALINDIA', 'DRREDDY', 'GAIL', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HEROMOTOCO', 
             'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY',  'KOTAKBANK',
            'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'RELIANCE', 'SBIN', 'SUNPHARMA',
            'TCS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN', 'UPL',  'WIPRO']
            
gan_static_levels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21, 23, 25, 28, 31, 34, 37, 40, 43, 46, 49, 53, 57, 61, 65, 69, 73, 77, 81, 86, 91, 96, 101, 106, 111, 116, 121, 127, 133, 139, 145, 151, 157, 163, 169, 176, 183, 190, 197, 204, 211, 218, 225, 233, 241, 249, 257, 265, 273, 281, 289, 298, 307, 316, 325, 334, 343, 352, 361, 371, 381, 391, 401, 411, 421, 431, 441, 452, 463, 474, 485, 496, 507, 518, 529, 541, 553, 565, 577, 589, 601, 613, 625, 638, 651, 664, 677, 690, 703, 716, 729, 743, 757, 771, 785, 799, 813, 827, 841, 856, 871, 886, 901, 916, 931, 946, 961, 977, 993, 1009, 1025, 1041, 1057, 1073, 1089, 1106, 1123, 1140, 1157, 1174, 1191, 1208, 1225, 1243, 1261, 1279, 1297, 1315, 1333, 1351, 1369, 1388, 1407, 1426, 1445, 1464, 1483, 1502, 1521, 1541, 1561, 1581, 1601, 1621, 1641, 1661, 1681, 1702, 1723, 1744, 1765, 1786, 1807, 1828, 1849, 1871, 1893, 1915, 1937, 1959, 1981, 2003, 2025, 2048, 2071, 2094, 2117, 2140, 2163, 2186, 2209, 2233, 2257, 2281, 2305, 2329, 2353, 2377, 2401, 2426, 2451, 2476, 2501, 2526, 2551, 2576, 2601, 2627, 2653, 2679, 2705, 2731, 2757, 2783, 2809, 2836, 2863, 2890, 2917, 2944, 2971, 2998, 3025, 3053, 3081, 3109, 3137, 3165, 3193, 3221, 3249, 3278, 3307, 3336, 3365, 3394, 3423, 3452, 3481, 3511, 3541, 3571, 3601, 3631, 3661, 3691, 3721, 3752, 3783, 3814, 3845, 3876, 3907, 3938, 3969, 4001, 4033, 4065, 4097, 4129, 4161, 4193, 4225, 4258, 4291, 4324, 4357, 4390, 4423, 4456, 4489, 4523, 4557, 4591, 4625, 4659, 4693, 4727, 4761, 4796, 4831, 4866, 4901, 4936, 4971, 5006, 5041, 5077, 5113, 5149, 5185, 5221, 5257, 5293, 5329, 5366, 5403, 5440, 5477, 5514, 5551, 5588, 5625, 5663, 5701, 5739, 5777, 5815, 5853, 5891, 5929, 5968, 6007, 6046, 6085, 6124, 6163, 6202, 6241, 6281, 6321, 6361, 6401, 6441, 6481, 6521, 6561, 6602, 6643, 6684, 6725, 6766, 6807, 6848, 6889, 6931, 6973, 7015, 7057, 7099, 7141, 7183, 7225, 7268, 7311, 7354, 7397, 7440, 7483, 7526, 7569, 7613, 7657, 7701, 7745, 7789, 7833, 7877, 7921, 7966, 8011, 8056, 8101, 8146, 8191, 8236, 8281, 8327, 8373, 8419, 8465, 8511, 8557, 8603, 8649, 8696, 8743, 8790, 8837, 8884, 8931, 8978, 9025, 9073, 9121, 9169, 9217, 9265, 9313, 9361, 9409, 9458, 9507, 9556, 9605, 9654, 9703, 9752, 9801, 9851, 9901, 9951, 10001, 10051, 10101, 10151, 10201, 10252, 10303, 10354, 10405, 10456, 10507, 10558, 10609, 10661, 10713, 10765, 10817, 10869, 10921, 10973, 11025, 11078, 11131, 11184, 11237, 11290, 11343, 11396, 11449, 11503, 11557, 11611, 11665, 11719, 11773, 11827, 11881, 11936, 11991, 12046, 12101, 12156, 12211, 12266, 12321, 12377, 12433, 12489, 12545, 12601, 12657, 12713, 12769, 12826, 12883, 12940, 12997, 13054, 13111, 13168, 13225, 13283, 13341, 13399, 13457, 13515, 13573, 13631, 13689, 13748, 13807, 13866, 13925, 13984, 14043, 14102, 14161, 14221, 14281, 14341, 14401, 14461, 14521, 14581, 14641, 14702, 14763, 14824, 14885, 14946, 15007, 15068, 15129, 15191, 15253, 15315, 15377, 15439, 15501, 15563, 15625, 15688, 15751, 15814, 15877, 15940, 16003, 16066, 16129, 16193, 16257, 16321, 16385, 16449, 16513, 16577, 16641, 16706, 16771, 16836, 16901, 16966, 17031, 17096, 17161, 17227, 17293, 17359, 17425, 17491, 17557, 17623, 17689, 17756, 17823, 17890, 17957, 18024, 18091, 18158, 18225, 18293, 18361, 18429, 18497, 18565, 18633, 18701, 18769, 18838, 18907, 18976, 19045, 19114, 19183, 19252, 19321, 19391, 19461, 19531, 19601, 19671, 19741, 19811, 19881, 19952, 20023, 20094, 20165, 20236, 20307, 20378, 20449, 20521, 20593, 20665, 20737, 20809, 20881, 20953, 21025, 21098, 21171, 21244, 21317, 21390, 21463, 21536, 21609, 21683, 21757, 21831, 21905, 21979, 22053, 22127, 22201, 22276, 22351, 22426, 22501, 22576, 22651, 22726, 22801, 22877, 22953, 23029, 23105, 23181, 23257, 23333, 23409, 23486, 23563, 23640, 23717, 23794, 23871, 23948, 24025, 24103, 24181, 24259, 24337, 24415, 24493, 24571, 24649, 24728, 24807, 24886, 24965, 25044, 25123, 25202, 25281, 25361, 25441, 25521, 25601, 25681, 25761, 25841, 25921, 26002, 26083, 26164, 26245, 26326, 26407, 26488, 26569, 26651, 26733, 26815, 26897, 26979, 27061, 27143, 27225, 27308, 27391, 27474, 27557, 27640, 27723, 27806, 27889, 27973, 28057, 28141, 28225, 28309, 28393, 28477, 28561, 28646, 28731, 28816, 28901, 28986, 29071, 29156, 29241, 29327, 29413, 29499, 29585, 29671, 29757, 29843, 29929, 30016, 30103, 30190, 30277, 30364, 30451, 30538, 30625, 30713, 30801, 30889, 30977, 31065, 31153, 31241, 31329, 31418, 31507, 31596, 31685, 31774, 31863, 31952, 32041, 32131, 32221, 32311, 32401, 32491, 32581, 32671, 32761, 32852, 32943, 33034, 33125, 33216, 33307, 33398, 33489, 33581, 33673, 33765, 33857, 33949, 34041, 34133, 34225, 34318, 34411, 34504, 34597, 34690, 34783, 34876, 34969, 35063, 35157, 35251, 35345, 35439, 35533, 35627, 35721, 35816, 35911, 36006, 36101, 36196, 36291, 36386, 36481, 36577, 36673, 36769, 36865, 36961, 37057, 37153, 37249, 37346, 37443, 37540, 37637, 37734, 37831, 37928, 38025, 38123, 38221, 38319, 38417, 38515, 38613, 38711, 38809]

def get_extrema_x(isMin,d1,d2,h):
  return [x for x in range(len(d1))
    if (d2[x] > 0 if isMin else d2[x] < 0) and
      (d1[x] == 0 or #slope is 0
        (x != len(d1) - 1 and #check next day
          (d1[x] > 0 and d1[x+1] < 0 and
           h[x] >= h[x+1] or
           d1[x] < 0 and d1[x+1] > 0 and
           h[x] <= h[x+1]) or
         x != 0 and #check prior day
          (d1[x-1] > 0 and d1[x] < 0 and
           h[x-1] < h[x] or
           d1[x-1] < 0 and d1[x] > 0 and
           h[x-1] > h[x])))]

def get_extrema_y(isMin,d1,d2,h):
  return [h[x] for x in range(len(d1))
    if (d2[x] > 0 if isMin else d2[x] < 0) and
      (d1[x] == 0 or #slope is 0
        (x != len(d1) - 1 and #check next day
          (d1[x] > 0 and d1[x+1] < 0 and
           h[x] >= h[x+1] or
           d1[x] < 0 and d1[x+1] > 0 and
           h[x] <= h[x+1]) or
         x != 0 and #check prior day
          (d1[x-1] > 0 and d1[x] < 0 and
           h[x-1] < h[x] or
           d1[x-1] < 0 and d1[x] > 0 and
           h[x-1] > h[x])))]
           
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

def bullish_divergence(data):
    

    h=data["Close"]
    rsi=data["RSI"]

    dx=1 #interval always fixed

    d_dx=FinDiff(0,dx,1) #first differntial
    d1=d_dx(h)

    d2_dx=FinDiff(0,dx,2) #second differntial
    d2=d2_dx(h)

    prices_min=get_extrema_y(True,d1,d2,h)
    prices_min_index=get_extrema_x(True,d1,d2,h)
    
    datetime_list=[]
    
    if(h[len(h)-2]==prices_min[len(prices_min)-2]):
        current_index=len(h)-2
        current_price=h[len(h)-2]
        current_rsi=rsi[len(rsi)-2]

        for i in range(0,len(prices_min)-1):
            if(prices_min[i]>current_price and current_rsi>rsi[prices_min_index[i]]):
                start=data["DateTime"][prices_min_index[i]]
                end=data["DateTime"][current_index]
                datetime_list.append([start,end])
                
    return (datetime_list)

def bearish_divergence(data):
    

    h=data["Close"]
    rsi=data["RSI"]

    dx=1 #interval always fixed

    d_dx=FinDiff(0,dx,1) #first differntial
    d1=d_dx(h)

    d2_dx=FinDiff(0,dx,2) #second differntial
    d2=d2_dx(h)

    prices_max=get_extrema_y(False,d1,d2,h)
    prices_max_index=get_extrema_x(False,d1,d2,h)
    
    datetime_list=[]
    
    if(h[len(h)-2]==prices_max[len(prices_max)-2]):
        current_index=len(h)-2
        current_price=h[len(h)-2]
        current_rsi=rsi[len(rsi)-2]

        for i in range(0,len(prices_max)-1):
            if(prices_max[i]<current_price and current_rsi<rsi[prices_max_index[i]]):
                start=data["DateTime"][prices_min_index[i]]
                end=data["DateTime"][current_index]
                datetime_list.append([start,end])
                
    return (datetime_list)
    
def bullish_hidden_divergence(data):
    

    h=data["Close"]
    rsi=data["RSI"]

    dx=1 #interval always fixed

    d_dx=FinDiff(0,dx,1) #first differntial
    d1=d_dx(h)

    d2_dx=FinDiff(0,dx,2) #second differntial
    d2=d2_dx(h)

    prices_min=get_extrema_y(True,d1,d2,h)
    prices_min_index=get_extrema_x(True,d1,d2,h)
    
    datetime_list=[]
    
    if(h[len(h)-2]==prices_min[len(prices_min)-2]):
        current_index=len(h)-2
        current_price=h[len(h)-2]
        current_rsi=rsi[len(rsi)-2]

        for i in range(0,len(prices_min)-1):
            if(prices_min[i]<current_price and current_rsi<rsi[prices_min_index[i]]):
                start=data["DateTime"][prices_min_index[i]]
                end=data["DateTime"][current_index]
                datetime_list.append([start,end])
                
    return (datetime_list)

def bearish_hidden_divergence(data):
    

    h=data["Close"]
    rsi=data["RSI"]

    dx=1 #interval always fixed

    d_dx=FinDiff(0,dx,1) #first differntial
    d1=d_dx(h)

    d2_dx=FinDiff(0,dx,2) #second differntial
    d2=d2_dx(h)

    prices_max=get_extrema_y(False,d1,d2,h)
    prices_max_index=get_extrema_x(False,d1,d2,h)
    
    datetime_list=[]
    
    if(h[len(h)-2]==prices_max[len(prices_max)-2]):
        current_index=len(h)-2
        current_price=h[len(h)-2]
        current_rsi=rsi[len(rsi)-2]

        for i in range(0,len(prices_max)-1):
            if(prices_max[i]<current_price and current_rsi<rsi[prices_max_index[i]]):
                start=data["DateTime"][prices_min_index[i]]
                end=data["DateTime"][current_index]
                datetime_list.append([start,end])
                
    return (datetime_list)
    
stock={"Symbol":[],"Strategy":[],"Entry":[],"StopLoss":[],"Target1":[],"Target2":[],"Target3":[]}
stock=pd.DataFrame(stock)

for i in Nifty50:
    
    data = td_app.get_historic_data(i, duration='4 D', bar_size='15 min')
    #symbol=i+".csv"
    #data=pd.read_csv("TATAMOTORS.csv")
    data=pd.DataFrame(data)
    data.columns=["Time","Open","High","Low","Close","Volume","oi"]
    data=data[["Time","Open","High","Low","Close","Volume"]]

    data["DateTime"]=data["Time"]
    data["Time"]=pd.to_datetime(data["DateTime"],format="%Y-%m-%d").dt.time
    data["Date"]=pd.to_datetime(data["DateTime"],format="%Y-%m-%d").dt.date
    
    filt1=(data["Time"]!=datetime.strptime("09:00:00", "%H:%M:%S").time()) 
    filt2=(data["Time"]!=datetime.strptime("15:30:00", "%H:%M:%S").time())
    filt3=(data["Time"]!=datetime.strptime("16:00:00", "%H:%M:%S").time())
    data["Time"]=data.loc[filt1,"Time"]
    data["Time"]=data.loc[filt2,"Time"]
    data["Time"]=data.loc[filt3,"Time"]
    data.dropna(inplace=True)
    
    
    data.reset_index(inplace=True)
    rows,columns=data.shape
    data=RSI(data,14)
    
    bullish_divergence_list=bullish_divergence(data)
    bearish_divergence_list=bearish_divergence(data)
    bullish_hidden_divergence_list=bullish_hidden_divergence(data)
    bearish_hidden_divergence_list=bearish_hidden_divergence(data)

    if(len(bullish_divergence_list)!=0):
        print(bullish_divergence_list)
    if(len(bearish_divergence_list)!=0):    
        print(bearish_divergence_list)
    if(len(bullish_hidden_divergence_list)!=0):    
        print(bullish_hidden_divergence_list)
    if(len(bullish_divergence_list)!=0):
        print(bearish_hidden_divergence_list)