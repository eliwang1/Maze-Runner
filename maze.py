import numpy as np
import random

FT = []
R = []
Q = []

def CreateMaze(ns):
    for i in range(ns):
        FT.append([0] * ns)
    FT[0][1] = FT[0][4] = FT[1][0] = FT[1][5] = FT[2][3] = 1
    FT[2][6] = FT[3][2] = FT[3][7] = FT[4][0] = FT[4][8] = 1
    FT[5][1] = FT[5][6] = FT[5][9] = FT[6][2] = FT[6][5] = 1
    FT[6][7] = FT[7][3] = FT[7][6] = FT[7][11] = FT[8][4] = 1
    FT[8][9] = FT[9][5] = FT[9][8] = FT[9][10] = FT[10][9] = 1
    FT[11][11] = 1;  # goal
    return FT

def CreateReward(ns):
    for i in range(ns):
        R.append([0] * ns)
    R[0][1] = R[0][4] = R[1][0] = R[1][5] = R[2][3] = -0.1;
    R[2][6] = R[3][2] = R[3][7] = R[4][0] = R[4][8] = -0.1;
    R[5][1] = R[5][6] = R[5][9] = R[6][2] = R[6][5] = -0.1;
    R[6][7] = R[7][3] = R[7][6] = R[7][11] = R[8][4] = -0.1;
    R[8][9] = R[9][5] = R[9][8] = R[9][10] = R[10][9] = -0.1;
    R[7][11] = 10.0;  # goal
    return R

def CreateQuality(ns):
    for i in range(ns):
        Q.append([0] * ns)
    return Q

def GetPossNextStates(s, FT):
    result = []
    for j in range(len(FT)):
        if FT[s][j] == 1:
            result.append(j)
    return result

def GetRandNextState(s, FT):
      possNextStates = GetPossNextStates(s, FT)
      ct = len(possNextStates) - 1
      idx = random.randint(0, ct)
      return possNextStates[idx]

def Train( FT, R, Q, goal, gamma, lrnRate, maxEpochs):
    for epoch in range(maxEpochs):
        currState = random.randint(0, len(R) - 1)
        while True:
            nextState = GetRandNextState(currState, FT)
            possNextNextStates = GetPossNextStates(nextState, FT)
            maxQ = -1.7976931348623157E+308 # double.MinValue;
            for j in range(len(possNextNextStates)):
                nns = possNextNextStates[j];  # short alias
                q = Q[nextState][nns]
                if q > maxQ:
                    maxQ = q

            # Q = [(1-a) * Q]  +  [a * (rt + (g * maxQ))]
            Q[currState][nextState] = ((1 - lrnRate) * Q[currState][nextState]) + (lrnRate * (R[currState][nextState] + (gamma * maxQ)))

            currState = nextState
            if currState == goal:
                break
# Train

def Print(Q):
    ns = len(Q)
    print("        [0]    [1]    [2]    [3]    [4]");
    print("    [5]    [6]    [7]    [8]    [9]");
    print("   [10]   [11]");
    for i in range(ns):
        row = "[" + str(i) + "]";
        print(row);
        for j in range(ns):
            s = str(Q[i][j])
            if s.find("-", 0, len(s)) == False:
                print(" ", s) # .PadLeft(6))
            else:
                print(s) # .PadLeft(7));
#        Console.WriteLine();

def Walk(start, goal, Q):
    curr = start
    next1 = str(curr) + "->"    
    while curr != goal:
      next = ArgMax(Q[curr])
      next1 = next1 + str(next) + "->"
      curr = next
    print(next1)
    print("done")

def ArgMax(vector):
    maxVal = vector[0]
    idx = 0
    for i in range(len(vector)):
        if vector[i] > maxVal:
          maxVal = vector[i]
          idx = i
    return idx

print("\nBegin maze RL Q-learning demo \n")
print("Setting up maze and rewards \n")
ns = 12
FT = CreateMaze(ns)
R = CreateReward(ns)
Q = CreateQuality(ns)
currState = random.randint(0, len(R) - 1)
s = currState
possNextStates = GetPossNextStates(s, FT)
# print(possNextStates)
print("Analyzing maze using Q-learning ")
goal = 11
gamma = 0.5
learnRate = 0.5
maxEpochs = 1000
Train(FT, R, Q, goal, gamma, learnRate, maxEpochs)

print("Done. Q matrix: \n ")
for i in range(ns):
    line1 = ""
    for j in range(ns):
        len1 = len(str(np.round(Q[i][j], decimals=2)))
        while len1 < 8:
            line1 = line1 + " "
            len1 += 1
        line1 += str(np.round(Q[i][j], decimals=2));
    print(line1)

print("\nUsing Q to walk from cell 8 to 11");
Walk(8, 11, Q);
print("\nEnd demo ");
# Console.ReadLine();
# Main
