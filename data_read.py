# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:33:30 2023

@author: alier
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

 


years = np.arange(2000,2020)
day = np.array([7,14,21,28])
mon = np.arange(1,13)
comps = ["PM25","CO","O3","NO","NO2"]


### PM25

df_PM25 = pd.DataFrame()

for y in years:
    for m in mon:
        for d in day:
            
                
                dft = pd.read_csv("data/mdata"+str(y)+"-"+str(m)+"-"+str(d)+"-PM25.txt")
                
                
                df_PM25 = pd.concat([df_PM25,dft])
                
                print(str(y)+"-"+str(m)+"-"+str(d)+"-PM25")

# df_PM25.head()
# df_PM25.dtypes
df_PM25=df_PM25.iloc[:,[0,1,3]]

df_PM25 = df_PM25.rename(columns={"pm25_davg": "value"})

polutant = np.full(df_PM25.shape[0],"PM25")

df_PM25['polutant'] = polutant

df_PM25.to_csv("data/afPM25.txt", index=False)


### CO

df_CO = pd.DataFrame()

for y in years:
    for m in mon:
        for d in day:
            
                
                dft = pd.read_csv("data/mdata"+str(y)+"-"+str(m)+"-"+str(d)+"-CO.txt")
                
                
                df_CO = pd.concat([df_CO,dft])
                
                print(str(y)+"-"+str(m)+"-"+str(d)+"-CO")
                
# df_CO.head()
# df_CO.dtypes
df_CO=df_CO.iloc[:,[0,1,2]]

df_CO = df_CO.rename(columns={"co_davg": "value"})

polutant = np.full(df_CO.shape[0],"CO")

df_CO['polutant'] = polutant

df_CO.to_csv("data/afCO.txt", index=False)


### O3

df_O3 = pd.DataFrame()
                
for y in years:
    for m in mon:
        for d in day:
            
                
                dft = pd.read_csv("data/mdata"+str(y)+"-"+str(m)+"-"+str(d)+"-O3.txt")
                
                
                df_O3 = pd.concat([df_O3,dft])
                
                print(str(y)+"-"+str(m)+"-"+str(d)+"-O3")
                
# df_O3.head()
# df_O3.dtypes
df_O3=df_O3.iloc[:,[0,1,2]]

df_O3 = df_O3.rename(columns={"ozone_davg": "value"})

polutant = np.full(df_O3.shape[0],"O3")

df_O3['polutant'] = polutant

df_O3.to_csv("data/afO3.txt", index=False)


### NO

df_NO = pd.DataFrame()
                
for y in years:
    for m in mon:
        for d in day:
            
                
                dft = pd.read_csv("data/mdata"+str(y)+"-"+str(m)+"-"+str(d)+"-NO.txt")
                
                
                df_NO = pd.concat([df_NO,dft])
                
                print(str(y)+"-"+str(m)+"-"+str(d)+"-NO")

# df_NO.head()
# df_NO.dtypes
df_NO=df_NO.iloc[:,[0,1,2]]

df_NO = df_NO.rename(columns={"no_davg": "value"})

polutant = np.full(df_NO.shape[0],"NO")

df_NO['polutant'] = polutant

df_NO.to_csv("data/afNO.txt", index=False)


### NO2

df_NO2 = pd.DataFrame()               

for y in years:
    for m in mon:
        for d in day:
            
                
                dft = pd.read_csv("data/mdata"+str(y)+"-"+str(m)+"-"+str(d)+"-NO2.txt")
                
                
                df_NO2 = pd.concat([df_NO2,dft])
                
                print(str(y)+"-"+str(m)+"-"+str(d)+"-NO2")

# df_NO2.head()
# df_NO2.dtypes
df_NO2=df_NO2.iloc[:,[0,1,2]]

df_NO2 = df_NO2.rename(columns={"no2_davg": "value"})

polutant = np.full(df_NO2.shape[0],"NO2")

df_NO2['polutant'] = polutant

df_NO2.to_csv("data/afNO2.txt", index=False)


#### concatenate all datasets


df_all = pd.concat([df_PM25,df_CO,df_O3,df_NO,df_NO2])
df_all.to_csv("data/afALL.txt", index=False)

df_wide = pd.pivot_table(df_all, values="value", index=["summary_date", "site"], columns=["polutant"])
df_wide.to_csv("data/afWIDE.txt", index=False)


np.sum(df_wide["CO"].isna())
np.sum(df_wide["O3"].isna())
np.sum(df_wide["NO"].isna())
np.sum(df_wide["NO2"].isna())

np.sum(df_wide["CO"]<0)
np.sum(df_wide["O3"]<0)
np.sum(df_wide["NO"]<0)
np.sum(df_wide["NO2"]<0)

np.sum(df_wide["CO"]<=0)
np.sum(df_wide["O3"]<=0)
np.sum(df_wide["NO"]<=0)
np.sum(df_wide["NO2"]<=0)




###### dropping na

# df_all = pd.concat([df_CO,df_O3,df_NO,df_NO2])
# df_wide = pd.pivot_table(df_all, values="value", index=["summary_date", "site"], columns=["polutant"])
# df_nona = df_wide.dropna()



# grouped = df_nona.groupby(['summary_date']).count()
# # grouped.describe()
# # np.max(grouped)
# # grouped[np.argmax(grouped["CO"]):np.argmax(grouped["CO"])+1]
# perf_date = df_nona.loc[["2002-04-26"]]
# perf_date = (perf_date == 0)*0.0005 + perf_date
# # np.min(perf_date)
# # np.sum(perf_date<0)


# perf_date.to_csv("data/april2602Data.txt")


###### no dropping na

df_all = pd.concat([df_CO,df_O3,df_NO,df_NO2])
df_wide = pd.pivot_table(df_all, values="value", index=["summary_date", "site"], columns=["polutant"])




grouped = df_wide.groupby(['summary_date']).count()
# grouped.describe()
# np.max(grouped)
# grouped[np.argmax(grouped["CO"]):np.argmax(grouped["CO"])+1]
perf_date = df_wide.loc[["2002-04-26"]]
perf_date = (perf_date == 0)*0.0005 + perf_date
# np.min(perf_date)
# np.sum(perf_date<0)

### only have O3 measurements ###
ind = np.logical_not(perf_date["CO"].isna() & perf_date["NO"].isna() & perf_date["NO2"].isna())
np.sum(1-ind)

# ind = np.logical_not(perf_date["CO"].isna() & perf_date["NO"].isna() & perf_date["O3"].isna())
# np.sum(1-ind)

# ind = np.logical_not(perf_date["CO"].isna() & perf_date["NO2"].isna() & perf_date["O3"].isna())
# np.sum(1-ind)

# ind = np.logical_not(perf_date["NO"].isna() & perf_date["NO2"].isna() & perf_date["O3"].isna())
# np.sum(1-ind)

perf_date = perf_date[ind]


perf_date.to_csv("data/april2602DataMiss.txt")

####


april_data = pd.read_csv("data/april2602DataMiss.txt")
# np.round(np.cov(np.log(april_data.iloc[:,2:]),rowvar=False),2)

sites_CO = pd.read_csv("data/aCO_sites.txt")
sites_NO = pd.read_csv("data/aNO_sites.txt")
sites_NO2 = pd.read_csv("data/aNO2_sites.txt")
sites_O3 = pd.read_csv("data/aO3_sites.txt")

sites = pd.concat([sites_CO,sites_NO,sites_NO2,sites_O3])
sites.drop_duplicates(subset=['site'], keep="first", inplace=True)
sites = sites.sort_values(by="site")
# sites.columns
# sites.dtypes

right_sites = sites[np.isin(sites["site"],april_data["site"])]

lat = np.array(right_sites["latitude"])
long = np.array(right_sites["longitude"])

mid_lat = (np.max(lat) + np.min(lat))/2

x = long*np.cos(mid_lat/360*2*np.pi)
y = lat

range_x = np.max(x) - np.min(x)
range_y = np.max(y) - np.min(y)

x = (x - np.min(x))/range_y
y = (y - np.min(y))/range_y

# plt.scatter(long,lat)
# plt.show()

# fig, ax = plt.subplots()
# # ax.set_xlim(0,1)
# # ax.set_ylim(0,1)
# ax.set_box_aspect(1)

# plt.scatter(x,y)
# plt.show()

april_data['longitude'] = long
april_data['latitude'] = lat
april_data['x'] = x
april_data['y'] = y


april_data.to_csv("data/april2602xyDataMiss.txt")






