#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import os
import glob

imgs1 = []
label1 = []
value1 = []
imgs2 = []
label2 = []
value2 = []
for paths in os.listdir("./carpet"):
    img_path = os.listdir(os.path.join("./carpet/"+paths))
    for i in range(len(img_path)):
        #print(i)
        if i % 7 != 0:
            imgs1.append(paths + "/" + img_path[i])
            label1.append(paths.split("_")[-1])
            value1.append(1)
        else:
            imgs2.append(paths + "/" + img_path[i])
            label2.append(paths.split("_")[-1])
            value2.append(1)

df1 = pd.DataFrame()
df1["ID"] = imgs1
df1["Label"] = label1
df1['value'] = value1
df1 = df1.pivot(index='ID', columns='Label', values='value').reset_index()
df1.fillna(0, inplace=True)
print(df1.tail())

df2 = pd.DataFrame()
df2["ID"] = imgs2
df2["Label"] = label2
df2['value'] = value2
df2 = df2.pivot(index='ID', columns='Label', values='value').reset_index()
df2.fillna(0, inplace=True)
print(len(imgs1),len(imgs2))
df1.to_csv("./annos/train.csv",index=False)
df2.to_csv("./annos/val.csv",index=False)
