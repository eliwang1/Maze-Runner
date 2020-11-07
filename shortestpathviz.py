import numpy as np
import random
# fsm.py - http://www.graphviz.org/content/fsm
from graphviz import Digraph
import os

FT = []
R = []
RMax = 99
Q = []
V = []
E = []
S = []

def CreateMaze(ns):
    for i in range(ns):
        FT.append([0] * ns)
    FT[0][1] = FT[1][0] = 1
    FT[0][2] = FT[2][0] = 1
    FT[0][3] = FT[3][0] = 1
    FT[1][7] = FT[7][1] = 1
    FT[1][5] = FT[5][1] = 1
    FT[1][2] = FT[2][1] = 1
    FT[2][5] = FT[5][2] = 1
    FT[2][4] = FT[4][2] = 1
    FT[2][3] = FT[3][2] = 1
    FT[3][4] = FT[4][3] = 1
    FT[3][8] = FT[8][3] = 1
    FT[3][2] = FT[2][3] = 1
    FT[4][5] = FT[5][4] = 1
    FT[4][6] = FT[6][4] = 1
    FT[4][8] = 1  # goal
    FT[5][6] = FT[6][5] = 1
    FT[5][7] = FT[7][5] = 1
    FT[6][8] = 1  # goal
    FT[7][6] = FT[6][7] = 1
    FT[7][8] = 1  # goal
    return FT

def CreateValue(ns):
    for i in range(ns):
        V.append([0] * ns)
    V[0][1] = 15
    V[0][2] = 13
    V[0][3] = 5
    V[1][7] = 11
    V[1][5] = 8
    V[1][2] = V[2][1] = 2
    V[2][5] = V[5][2] = 6
    V[2][4] = V[4][2] = 3
    V[2][3] = V[3][2] = 18
    V[3][4] = V[4][3] = 4
    V[3][8] = V[8][3] = 99
    V[3][2] = V[2][3] = 18
    V[4][5] = V[5][4] = 1
    V[4][6] = V[6][4] = 9
    V[4][8] = 14 
    V[5][6] = V[6][5] = 16
    V[5][7] = V[7][5] = 17
    V[7][6] = V[6][7] = 7
    V[6][8] = 10 
    V[7][8] = 12 
    return V

def CreateReward(ns):
    for i in range(ns):
        R.append([0] * ns)
    R[0][1] = RMax - V[0][1]
    R[0][2] = RMax - V[0][2]
    R[0][3] = RMax - V[0][3]
    R[1][7] = RMax - V[1][7]
    R[1][5] = RMax - V[1][5]
    R[1][2] = R[2][1] = RMax - V[1][2]
    R[2][5] = R[5][2] = RMax - V[2][5]
    R[2][4] = R[4][2] = RMax - V[2][4]
    R[2][3] = R[3][2] = RMax - V[2][3]
    R[3][4] = R[4][3] = RMax - V[3][4]
    R[3][8] = R[8][3] = RMax - V[3][8]
    R[3][2] = R[2][3] = RMax - V[3][2]
    R[4][5] = R[5][4] = RMax - V[4][5]
    R[4][6] = R[6][4] = RMax - V[4][6]
    R[4][8] = RMax - V[4][8] + 100  # goal
    R[5][6] = R[6][5] = RMax - V[5][6]
    R[5][7] = R[7][5] = RMax - V[5][7]
    R[7][6] = R[6][7] = RMax - V[7][6]
    R[6][8] = RMax - V[6][8] + 100  # goal
    R[7][8] = RMax - V[7][8] + 100  # goal
    return R

def CreateQuality(ns):
    for i in range(ns):
        Q.append([0] * ns)
    return Q

def GetPossNextStates(s, FT):
    result = [];
    for j in range(len(FT)):
        if FT[s][j] == 1:
            result.append(j);
    return result;

def GetRandNextState(s, FT):
      possNextStates = GetPossNextStates(s, FT);
      ct = len(possNextStates) - 1;
      idx = random.randint(0, ct);
      return possNextStates[idx];

def Train( FT, R, Q, goal, gamma, lrnRate, maxEpochs):
    for epoch in range(maxEpochs):
        currState = random.randint(0, len(R) - 1)
        while True:
            nextState = GetRandNextState(currState, FT);
            possNextNextStates = GetPossNextStates(nextState, FT);
            maxQ = -1.7976931348623157E+308 # double.MinValue;
            for j in range(len(possNextNextStates)):
                nns = possNextNextStates[j];  # short alias
                q = Q[nextState][nns];
                if q > maxQ:
                    maxQ = q;

            # Q = [(1-a) * Q]  +  [a * (rt + (g * maxQ))]
            Q[currState][nextState] = ((1 - lrnRate) * Q[currState][nextState]) + (lrnRate * (R[currState][nextState] + (gamma * maxQ)));

            currState = nextState;
            if currState == goal:
                break;
# Train

def Walk(start, goal, Q):
    curr = start;
    next1 = str(curr) + "->"    
    attr1 = ""
    attr2 = "f.attr('node', shape='circle')\n"
    start1 = 1
    filled1 = ""
    filled2 = "f.attr('node', shape='circle', style='filled')\n"
    while curr != goal:
      next = ArgMax(Q[curr]);
      for i in range(start1, next):
        attr1 = attr2 + "f.node('LR_" + str(i) + "')\n"
        attr2 = attr1
      start1 = next + 1
      next1 = next1 + str(next) + "->"
      if next < (len(FT) - 1):
          filled1 = filled2 + "f.node('LR_" + str(next) + "')\n"
      filled2 = filled1
      curr = next;
    print(next1)
    print("done");
    return filled1, attr1

def ArgMax(vector):
    maxVal = vector[0];
    idx = 0;
    for i in range(len(vector)):
        if vector[i] > maxVal:
          maxVal = vector[i];
          idx = i;
    return idx;

print("\nBegin maze RL Q-learning demo \n")
print("Setting up maze and rewards \n")
ns = 9
FT = CreateMaze(ns)
V = CreateValue(ns)
R = CreateReward(ns)
Q = CreateQuality(ns)

currState = random.randint(0, len(R) - 1);
s = currState
possNextStates = GetPossNextStates(s, FT)
# print(possNextStates)
print("Analyzing maze using Q-learning ");
goal = 8;
gamma = 0.5;
learnRate = 0.5;
maxEpochs = 1000;
Train(FT, R, Q, goal, gamma, learnRate, maxEpochs);

print("Done. Q matrix: \n ");
print("          [0]     [1]     [2]     [3]     [4]     [5]     [6]     [7]     [8]");
for i in range(ns):
    line1 = "[" + str(i) + "]  "
    for j in range(ns):
        len1 = len(str(np.round(Q[i][j], decimals=2)))
        while len1 < 8:
            line1 = line1 + " "
            len1 += 1
        line1 += str(np.round(Q[i][j], decimals=2));
    print(line1)

print("\nUsing Q to walk from cell 0 to 8");
filled1, attr1 = Walk(0, 8, Q);
print("\nEnd demo ");
# Console.ReadLine();
# Main
os.remove("graph4.py")
spgraph = """ 
from graphviz import Digraph
f = Digraph('finite_state_machine', filename='ShortestPath.gv')
f.attr(rankdir='LR', size='8,5')

f.attr('node', shape='doublecircle')
f.node('LR_0')
""" 

h = open("graph4.py", "a")
h.write(spgraph)
h.write("f.node('LR_" + str(len(FT) - 1) + "')\n")
h.write(attr1)
h.write(filled1)

for i in range(ns):
	for j in range(ns):
	    if (FT[i][j] == 1 and i < j):
		    h.write("f.edge('LR_" + str(i) + "', 'LR_" + str(j) + "', label='" + str(V[i][j]) + "')\n")
h.write("f.view()")
h.close()
os.system("python graph4.py")
