# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 12:37:38 2023

@author: alier
"""

import numpy as np
import urllib.request

years = np.arange(2000,2020)
day = np.array([7,14,21,28])
mon = np.arange(1,13)


# ### PM25

# for y in years:
#     for m in mon:
#         for d in day:
    
#             print(str(y)+"-"+str(m)+"-"+str(d)+" PM25")
#             urllib.request.urlretrieve("https://www.arb.ca.gov/aqmis2/display.php?download=y&year="+str(y)+"&mon="+str(m)+"&day="+str(d)+"&param=PM25&units=001&statistic=DAVG&order=basin%2Ccounty_name%2Cname&county_name=--COUNTY--&basin=--AIR+BASIN--&latitude=A-Whole+State&std15=&o3switch=new&hours=all&ptype=aqd&report=7DAY&btnsubmit=Update+Display", "data/data"+str(y)+"-"+str(m)+"-"+str(d)+"-PM25.txt")

# ### CO

# for y in years:
#     for m in mon:
#         for d in day:
    
#             print(str(y)+"-"+str(m)+"-"+str(d)+" CO")
#             urllib.request.urlretrieve("https://www.arb.ca.gov/aqmis2/display.php?download=y&year="+str(y)+"&mon="+str(m)+"&day="+str(d)+"&param=CO&units=007&statistic=DAVG&order=basin%2Ccounty_name%2Cname&county_name=--COUNTY--&basin=--AIR+BASIN--&latitude=A-Whole+State&std15=&o3switch=new&hours=all&ptype=aqd&report=7DAY&btnsubmit=Update+Display", "data/data"+str(y)+"-"+str(m)+"-"+str(d)+"-CO.txt")

# ### O3

# for y in years:
#     for m in mon:
#         for d in day:
    
#             print(str(y)+"-"+str(m)+"-"+str(d)+" O3")
#             urllib.request.urlretrieve("https://www.arb.ca.gov/aqmis2/display.php?download=y&year="+str(y)+"&mon="+str(m)+"&day="+str(d)+"&param=OZONE_ppm&units=007&statistic=DAVG&order=basin%2Ccounty_name%2Cname&county_name=--COUNTY--&basin=--AIR+BASIN--&latitude=A-Whole+State&std15=&o3switch=new&hours=all&ptype=aqd&report=7DAY&btnsubmit=Update+Display", "data/data"+str(y)+"-"+str(m)+"-"+str(d)+"-O3.txt")


# ### NO

# for y in years:
#     for m in mon:
#         for d in day:
    
#             print(str(y)+"-"+str(m)+"-"+str(d)+" NO")
#             urllib.request.urlretrieve("https://www.arb.ca.gov/aqmis2/display.php?download=y&year="+str(y)+"&mon="+str(m)+"&day="+str(d)+"&param=NO&units=007&statistic=DAVG&order=basin%2Ccounty_name%2Cname&county_name=--COUNTY--&basin=--AIR+BASIN--&latitude=A-Whole+State&std15=&o3switch=new&hours=all&ptype=aqd&report=7DAY&btnsubmit=Update+Display", "data/data"+str(y)+"-"+str(m)+"-"+str(d)+"-NO.txt")

# ### NO2

# for y in years:
#     for m in mon:
#         for d in day:
    
#             print(str(y)+"-"+str(m)+"-"+str(d)+" NO2")
#             urllib.request.urlretrieve("https://www.arb.ca.gov/aqmis2/display.php?download=y&year="+str(y)+"&mon="+str(m)+"&day="+str(d)+"&param=NO2&units=007&statistic=DAVG&order=basin%2Ccounty_name%2Cname&county_name=--COUNTY--&basin=--AIR+BASIN--&latitude=A-Whole+State&std15=&o3switch=new&hours=all&ptype=aqd&report=7DAY&btnsubmit=Update+Display", "data/data"+str(y)+"-"+str(m)+"-"+str(d)+"-NO2.txt")



years = np.arange(2000,2020)
day = np.array([7,14,21,28])
mon = np.arange(1,13)
comps = ["PM25","CO","O3","NO","NO2"]



for y in years:
    for m in mon:
        for d in day:
            for c in comps:
                
                s = open("data/data"+str(y)+"-"+str(m)+"-"+str(d)+"-"+c+".txt")

                i=0
                for line in s:
                    
                    if line == " \n":
                        print(i)
                        break
                    i+=1



                s = open("data/data"+str(y)+"-"+str(m)+"-"+str(d)+"-"+c+".txt")
                lines = s.readlines()


                f = open("data/mdata"+str(y)+"-"+str(m)+"-"+str(d)+"-"+c+".txt", "w")
                for line in lines[:i]:
                    f.write(line)
                
                print(str(y)+"-"+str(m)+"-"+str(d)+"-"+c)
                












