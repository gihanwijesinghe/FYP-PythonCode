import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.cm as cmx
from matplotlib import cm
from itertools import combinations
import itertools as iter

# loading the csv file
#data = pd.read_csv('OverallPlayerPerformancesBidsNewUpdated.csv')
data = pd.read_csv('newPlayerPerformance.csv')

playerNames = data['Player_Name']
bidValue = data['bid_Value']

# for dropping multiple columns
df = data.drop(['Pos', 'Player_Name', 'Team', 'Team_Symbol', 'bid_Value'], axis=1)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(df)

#getting the centroids of the 3 clusters
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

#initializing the player groups and playerNo to include the index of the players
player1 = []
player2 = []
player3 = []
player1No = []
player2No = []
player3No = []
batsmanName = []
ballerName = []
allRounderName = []
batsmanNo = []
ballerNo = []
allRounderNo = []
#getting lable values and appending the values to groups
for num in range(0, len(data)):
    if labels[num] == 0:
        player1.append(playerNames[num])
        player1No.append(num)
    elif labels[num] == 1:
        player2.append(playerNames[num])
        player2No.append(num)
    else:
        player3.append(playerNames[num])
        player3No.append(num)

print(player1)
print(player2)
print(player3)

## naming the batsman, ballers and allrounders clusters
if player1[0] == 'Michael_Hussey':
    batsmanName = player1
    batsmanNo = player1No
    if player2[0] == 'Mitchell_McClenaghan':
        ballerName = player2
        ballerNo = player2No
        allRounderName = player3
        allRounderNo = player3No
    else:
        ballerName = player3
        ballerNo = player3No
        allRounderName = player2
elif player1[0] == 'Mitchell_McClenaghan':
    ballerName = player1
    ballerNo = player1No
    if player2[0] == 'Michael_Hussey':
        batsmanName = player2
        batsmanNo = player2No
        allRounderName = player3
        allRounderNo = player3No
    else:
        batsmanName = player3
        batsmanNo = player3No
        allRounderName = player2
        allRounderNo = player2No
else:
    allRounderName = player1
    allRounderNo = player1No
    if player2[0] == 'Mitchell_McClenaghan':
        ballerName = player2
        ballerNo = player2No
        batsmanName =player3
        batsmanNo = player3No
    else:
        ballerName = player3
        ballerNo = player3No
        batsmanName = player2
        batsmanNo = player2No

print(len(batsmanNo))
print(len(ballerNo))
print(len(allRounderNo))

#initializing the groups to group the players in each cluster
# batsmans - each group having 5
# bowlers - each group having 4
# allrounders - each group having two
player1Groups = []
player2Groups = []
player3Groups = []
players = []

def pset(lst, no):
    return [",".join(map(str, comb)) for comb in combinations(lst, no)]
    # comb = (iter.combinations(lst, l) for l in range(len(lst) + 1))
    # return list(iter.chain.from_iterable(comb))
# batsmanNo = [0, 1, 2, 3, 5, 6, 7]
# ballerNo = [21, 23, 25, 33, 44, 48]
# allRounderNo = [4, 15, 16, 22]

player1Groups = pset(batsmanNo, 5)
player2Groups = pset(ballerNo, 4)
player3Groups = pset(allRounderNo, 2)

## initializing the weights for each batsman, bowler and allrounder
player1Weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
player2Weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
player3Weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

#calculating the wiights and  values by muyltiplying weight vector with the values vector
def playerWinning(weights, values):
    sum = 0
    for num in range(0, len(weights)):
        sum = sum + (weights[num] * values[num])
    return sum

#claculating the sum of the weight* value for each group (eg:-5-batsman)
def allPlayersWinning(playerNos, weights, leng):
    sum =0
    for num in range(0, leng):
        intArray = playerNos.split(",")
        valueVector = df.iloc[int(intArray[num])].values
        sum = sum + playerWinning(weights, valueVector)
    return sum

#initializing the winnability vector and the x, y and the z vector
teams = []
winnability = []
xx = []
yy = []
zz = []

def appendPlayer(team, playerNos):
    for num in range(0, len(playerNos)):
        team.append(playerNames[int(playerNos[num])])
    return team

def formTeam(batsman, bowlers, allrounders, winValue, budget):
    teamPlayerNames = []
    batNos = batsman.split(",")
    bowlNos = bowlers.split(",")
    allRoNos = allrounders.split(",")
    teamPlayerNames =appendPlayer(teamPlayerNames, batNos)
    teamPlayerNames = appendPlayer(teamPlayerNames, bowlNos)
    teamPlayerNames = appendPlayer(teamPlayerNames, allRoNos)
    teamPlayerNames.append(winValue)
    teamPlayerNames.append(budget)
    return teamPlayerNames

df.columns = [''] * len(df.columns)#removing teh names of the columns

#printing CSV files
def printCSV(teamCombinations, fileNo):
    ddf = pd.DataFrame(teamCombinations)
    filename = 'composition/teams' + str(fileNo) + '.csv'
    ddf.to_csv(filename, index=False, encoding='utf-8')

#calculating the budget for group of plauers
def calBudget(players, leng):
    budget = 0
    playerNos = players.split(",")
    for num in range(0, leng):
        budget = budget + bidValue[int(playerNos[num])]
    return budget

#calculating the winnability and assigning them
i = 0
for p1 in range(0, len(player1Groups)):
    player1Sum = allPlayersWinning(player1Groups[p1], player1Weights, 5)
    budget1Sum = calBudget(player1Groups[p1], 5)
    for p2 in range(0, len(player2Groups)):
        player2Sum = allPlayersWinning(player2Groups[p2], player2Weights, 4)
        budget2Sum = calBudget(player2Groups[p2], 4)
        for p3 in range(0, len(player3Groups)):
            overallSum = player1Sum + player2Sum + allPlayersWinning(player3Groups[p3], player3Weights, 2)
            fullBudget = budget1Sum + budget2Sum + calBudget(player3Groups[p3], 2)
            xx.append(p1)
            yy.append(p2)
            zz.append(p3)
            winnability.append(overallSum)
            teams.append(formTeam(player1Groups[p1], player2Groups[p2], player3Groups[p3], overallSum, fullBudget))
            if (len(teams) == 1000):
                printCSV(teams, i)
                i = i +1
                teams = []
printCSV(teams, i)


#plotting the 4D plot
def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap,label='WinningPerformance')
    ax.set_xlabel('Batsmans')
    ax.set_ylabel('Bowlers')
    ax.set_zlabel('AllRounders')
    plt.show()

scatter3d(xx,yy,zz,winnability)
print(winnability)
print(teams)

