from __future__ import print_function
from util import Schedule

TTSA = Schedule(6,hardcoded = True)
TTSA.simulatedAnnealing()

TTSA = Schedule(6,hardcoded = True,maxR = 10,maxP = 100,maxC = 10)
TTSA.simulatedAnnealing()

TTSA = Schedule(6,hardcoded = True,maxR = 10,maxP = 100,maxC = 100)
TTSA.simulatedAnnealing()

TTSA = Schedule(6,hardcoded = True,maxR = 10,maxP = 100,maxC = 100)
TTSA.simulatedAnnealing()

TTSA = Schedule(6,hardcoded = True,maxR = 100,maxP = 100,maxC = 100)
TTSA.simulatedAnnealing()

'''
print(TTSA.distanceMap)
print(TTSA.cost())
print(TTSA.getViolations())
TTSA.partialSwapRounds(2,2,5)
print(TTSA.scheduleMap)
'''
