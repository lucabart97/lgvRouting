import os
import pandas as pd
import matplotlib.pyplot as plt

dictTime = {}
dictCost = {}
names = {}
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
            names[s[0]] = 0
            if not s[0] in dictTime:
                dictCost[s[0]] = list()
                dictTime[s[0]] = list()
            dictCost[s[0]].append(float(s[1]))
            dictTime[s[0]].append(float(s[2])/1000000)
#ordering
dfcost = pd.DataFrame(dictCost).T
dfcost = dfcost.sort_values(by=dfcost.columns[1]).T
dftime = pd.DataFrame(dictTime).T
dftime = dftime.sort_values(by=dftime.columns[1]).T

#print(dftime.T.sort_values(dftime.columns[0]))

print("BoxPlot with sample: "+str(numSample),end="\n\n")

plt.figure(1)
dfcost.boxplot()
plt.figure(2)
dftime.boxplot()
plt.show()
