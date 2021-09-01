import os
import pandas as pd
import matplotlib.pyplot as plt

dictTime = {}
dictCost = {}
numSample = 0

directory = "."
for entry in os.scandir(directory):
    if entry.name.startswith("log") and entry.is_file():
        data = open(entry.name).read()
        if len(data) < 2:
            continue
        numSample += 1
        lines = data.splitlines()
        for l in lines:
            s = l.split(";")
            if not s[0] in dictTime:
                dictCost[s[0]] = list()
                dictTime[s[0]] = list()
            dictCost[s[0]].append(float(s[1])/1000000)
            dictTime[s[0]].append(float(s[2])/1000000)

print("BoxPlot with sample: "+str(numSample))
dfcost = pd.DataFrame(dictCost)
dftime = pd.DataFrame(dictTime)

plt.figure(1)
dfcost.boxplot()
plt.figure(2)
dftime.boxplot()
plt.show()