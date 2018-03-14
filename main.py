from __future__ import print_function
from util import Schedule

TTSA = Schedule(6,hardcoded=True)
print(TTSA.scheduleMap)
TTSA.swapTeams(2,3)
print(TTSA.scheduleMap)

'''
print(TTSA.distanceMap)
print(TTSA.cost())
print(TTSA.getViolations())
TTSA.partialSwapRounds(2,2,5)
print(TTSA.scheduleMap)
'''
