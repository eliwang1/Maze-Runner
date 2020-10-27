import numpy as np
import random
# fsm.py - http://www.graphviz.org/content/fsm

from graphviz import Digraph

f = Digraph('finite_state_machine', filename='ShortestPath.gv')
f.attr(rankdir='LR', size='8,5')

f.attr('node', shape='doublecircle')
f.node('LR_0')
f.node('LR_8')

f.attr('node', shape='circle')
f.edge('LR_0', 'LR_1', label='15')
f.edge('LR_0', 'LR_2', label='13')
f.edge('LR_0', 'LR_3', label='5')
f.edge('LR_1', 'LR_7', label='11')
f.edge('LR_1', 'LR_5', label='8')
f.edge('LR_1', 'LR_2', label='2')
f.edge('LR_2', 'LR_3', label='18')
f.edge('LR_2', 'LR_5', label='6')
f.edge('LR_2', 'LR_4', label='3')
f.edge('LR_3', 'LR_2', label='18')
f.edge('LR_3', 'LR_4', label='4')
f.edge('LR_3', 'LR_8', label='99')
f.edge('LR_4', 'LR_5', label='1')
f.edge('LR_4', 'LR_6', label='9')
f.edge('LR_4', 'LR_8', label='14')
f.edge('LR_5', 'LR_7', label='17')
f.edge('LR_5', 'LR_6', label='16')
f.edge('LR_6', 'LR_8', label='10')
f.edge('LR_7', 'LR_8', label='12')
f.edge('LR_7', 'LR_6', label='7')

f.view()

FT = []
R = []
Q = []

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

def CreateReward(ns):
    for i in range(ns):
        R.append([0] * ns)
    R[0][1] = 99 - 15
    R[0][2] = 99 - 13
    R[0][3] = 99 - 5
    R[1][7] = 99 - 11
    R[1][5] = 99 - 8
    R[1][2] = R[2][1] = 99 - 2
    R[2][5] = R[5][2] = 99 - 6
    R[2][4] = R[4][2] = 99 - 3
    R[2][3] = R[3][2] = 99 - 18
    R[3][4] = R[4][3] = 99 - 4
    R[3][8] = R[8][3] = 99 - 99
    R[3][2] = R[2][3] = 99 - 18
    R[4][5] = R[5][4] = 99 - 1
    R[4][6] = R[6][4] = 99 - 9
    R[4][8] = 99 - 14 + 100  # goal
    R[5][6] = R[6][5] = 99 - 16
    R[5][7] = R[7][5] = 99 - 17
    R[7][6] = R[6][7] = 99 - 7
    R[6][8] = 99 - 10 + 100  # goal
    R[7][8] = 99 - 12 + 100  # goal
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
    while curr != goal:
      next = ArgMax(Q[curr]);
      next1 = next1 + str(next) + "->"
      curr = next;
    print(next1)
    print("done");

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
Walk(0, 8, Q);
print("\nEnd demo ");
# Console.ReadLine();
# Main
