import pandas as pd
# loading the csv file
#data = pd.read_csv("OverallPlayerPerformancesBidsNewUpdated.csv")
oridata = pd.read_csv('2017performancesCopy.csv')
data    = pd.read_csv('2017performancesCopy.csv')

playerNameIn = data['Player_Name_in']
playerName = data['Player_Name']
#print(len(playerNameIn))
print(len(oridata))


def delRow(row, cot):
    oridata.drop(oridata.index[row - cot], inplace=True)

i=0
count =0
for num in range(0, len(playerName)):
    for p in range(0, len(playerNameIn)):
        if playerName[num] == playerNameIn[p]:
            i = 1
    if i == 0:
        print(num)
        delRow(num, count)
        count = count + 1
    i=0

print(oridata)
oridata.to_csv('111players.csv', index=False, encoding='utf-8')