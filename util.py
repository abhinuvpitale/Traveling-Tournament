from __future__ import print_function
import numpy as np
import random
import config
import hardcode
import math



class Schedule():
    # Works
    def __init__(self,n,hardcoded=False):
        self.n = n
        self.nTeams = n
        self.nRounds = 2*n - 2

        if hardcoded:
            self.scheduleMap, self.distanceMap = self.hardcode(6)
        else:
            self.scheduleMap = self.buildRandomSchedule()
            self.distanceMap = self.createDistanceMap()
        self.simulatedAnnealing()
        print(1)

    def hardcode(self,n):
        if n == 4:
            return hardcode.hardcode4
        if n == 6:
            return hardcode.hardcode6,hardcode.cost6
    # Generates a Random Schedule satisfying the hard constraints and one of the soft constraints
    def buildRandomSchedule(self):
        S = (self.n+1)*np.ones([self.n,2*self.n-2],dtype=int)
        return self.buildSchedule(S,0,0)


    # Back Tracking to build the schedule
    def buildSchedule(self,S,team,roundN):
        # Return if complete
        if self.checkComplete(S):
            return S

        # Get next round and team
        nextRound = roundN + 1
        nextTeam = team
        # Overflow
        if nextRound == self.nRounds:
            nextRound = 0
            nextTeam = nextTeam + 1

        # If exists, then go to the next round
        if S[team,roundN] != self.nTeams + 1:
            return self.buildSchedule(S,nextTeam,nextRound)

        # Find Q
        Q = self.getChoices(S,team,roundN)
        random.shuffle(Q)
        if Q is None:
            return None

        # Try games
        for q in Q:
            St = np.copy(S)
            St[team,roundN] = q
            St[abs(q)-1,roundN] = (team+1) * np.sign(q) * - 1
            Snext = self.buildSchedule(St,nextTeam,nextRound)
            if Snext is not None:
                return Snext

        return None


    # Helper for build Schedule to check if the the matrix is completely built
    def checkComplete(self,S):
        for idx in range(self.nTeams):
            for innerIdx in range(self.nRounds):
                if S[idx,innerIdx] == (self.nTeams + 1):
                    return False
        return True


    # Helper for build Schedule to get the Q matrix
    def getChoices(self,S,team,roundN):
        Q = []

        # All elements
        for item in range(1,self.nTeams+1):
            Q.append(item)
            Q.append(-item)

        # Get existing elements
        done = np.unique(S[team,:])
        for item in done:
            if item in Q:
                Q.remove(item)

        # Remove current team
        if team+1 in Q:
            Q.remove(team+1)
        if -(team+1) in Q:
            Q.remove(-(team+1))

        # Remove Past team
        if roundN > 0:
            if S[team,roundN-1] in Q:
                Q.remove(S[team,roundN-1])
            if -S[team,roundN-1] in Q:
                Q.remove(-S[team,roundN-1])

        # Remove teams in current round
        done = np.unique(S[:,roundN])
        for item in done:
            if item in Q:
                Q.remove(item)
            if -item in Q:
                Q.remove(-item)

        return Q


    # Creates a distance map between two teams, :TODO needs to be tested
    def createDistanceMap(self):
        distanceMap = np.zeros((self.n, self.n))
        for idx in range(self.n):
            for inneridx in range(idx + 1):
                dist = random.randint(config.minDist, config.maxDist)
                distanceMap[idx][inneridx] = dist
                distanceMap[inneridx][idx] = dist
        return distanceMap

    # Swap the home and away games of teamA and teamB
    def swapHomes(self,teamA,teamB):
        idxA = teamA - 1
        idxB = teamB - 1
        idx = np.where(abs(self.scheduleMap[idxA,:]) == teamB)
        idx1 = idx[0][0]
        idx2 = idx[0][1]
        temp = self.scheduleMap[idxA,idx1]
        self.scheduleMap[idxA,idx1] = self.scheduleMap[idxA,idx2]
        self.scheduleMap[idxA,idx2] = temp
        temp = np.copy(self.scheduleMap[idxB, idx1])
        self.scheduleMap[idxB, idx1] = self.scheduleMap[idxB, idx2]
        self.scheduleMap[idxB, idx2] = temp

    # Swap the rounds completely
    def swapRounds(self,roundA,roundB):
        roundA = roundA - 1
        roundB = roundB - 1
        temp = np.copy(self.scheduleMap[:,roundA])
        self.scheduleMap[:,roundA] = self.scheduleMap[:,roundB]
        self.scheduleMap[:,roundB] = temp

    # Swap Schedule of teams, i.e all games except home and away ones.
    def swapTeams(self,teamA,teamB):
        idxA = teamA - 1
        idxB = teamB - 1
        temp = np.copy(self.scheduleMap[idxA,:])
        self.scheduleMap[idxA,:] = self.scheduleMap[idxB,:]
        self.scheduleMap[idxB,:] = temp

        idx1 = np.where(abs(self.scheduleMap) == teamA)
        idx2 = np.where(abs(self.scheduleMap) == teamB)

        for element in range(len(idx1[0])):
            self.scheduleMap[idx1[0][element], idx1[1][element]] = int(np.sign(self.scheduleMap[idx1[0][element], idx1[1][element]])) * teamB

        for element in range(len(idx2[0])):
            self.scheduleMap[idx2[0][element], idx2[1][element]] = int(np.sign(self.scheduleMap[idx2[0][element], idx2[1][element]])) * teamA

        idx = np.where(abs(self.scheduleMap[idxA, :]) == teamB)
        idx0 = idx[0][0]
        idx1 = idx[0][1]
        self.scheduleMap[idxA][idx0] = -self.scheduleMap[idxA][idx0]
        self.scheduleMap[idxA][idx1] = -self.scheduleMap[idxA][idx1]

        idx = np.where(abs(self.scheduleMap[idxB, :]) == teamA)
        idx0 = idx[0][0]
        idx1 = idx[0][1]
        self.scheduleMap[idxB][idx0] = -self.scheduleMap[idxB][idx0]
        self.scheduleMap[idxB][idx1] = -self.scheduleMap[idxB][idx1]

    def partialSwapRounds(self,team,roundA,roundB):
        teamA = team
        swapArr = [teamA-1]
        while 1:
            for item in swapArr:
                if abs(self.scheduleMap[item,roundA-1])-1 not in swapArr:
                    swapArr.append(abs(self.scheduleMap[item,roundA-1])-1)
                if abs(self.scheduleMap[item,roundB-1])-1 not in swapArr:
                    swapArr.append(abs(self.scheduleMap[item,roundB-1])-1)

            if abs(self.scheduleMap[swapArr[-1],roundA-1])-1 in swapArr:
                if abs(self.scheduleMap[swapArr[-1],roundB-1])-1 in swapArr:
                    if abs(self.scheduleMap[swapArr[-2], roundA - 1])-1 in swapArr:
                        if abs(self.scheduleMap[swapArr[-2], roundB - 1])-1 in swapArr:
                            break

        for item in swapArr:
            temp1 = self.scheduleMap[item,roundA-1]
            temp2 = self.scheduleMap[item,roundB-1]
            self.scheduleMap[item,roundA-1] = temp2
            self.scheduleMap[item,roundB-1] = temp1

    def createDistanceMap(self):
        distanceMap = np.zeros((self.n,self.n))
        for idx in range(self.n):
            for inneridx in range(idx+1):
                dist = random.randint(config.minDist,config.maxDist)
                distanceMap[idx][inneridx] = dist
                distanceMap[inneridx][idx] = dist
        return distanceMap

    def cost(self, S):
        dist = list([0] * self.n)

        # Init Cost
        for idx in range(len(dist)):
            dist[idx] = self.getDist(S, idx + 1, idx + 1, S[idx, 0])


        # Intermediate Cost
        for roundN in range(1, (self.n * 2) - 2):
            for team in range(self.n):
                dist[team] = dist[team] + self.getDist(S, team + 1, S[team, roundN - 1], S[team][roundN])


        # Final Cost
        for idx in range(len(dist)):
            dist[idx] = dist[idx] + self.getDist(S, idx + 1, S[idx, -1], idx + 1)

        sum1 = 0
        for item in dist:
            sum1 = sum1 + item
        violations = self.getViolations(S)
        if violations > 0:
            thissum = self.complexCost(sum1, violations)
            print(thissum)
            return thissum
        else:
            print(sum1)
            return sum1

    def complexCost(self,sum1,violations):
        w = config.w
        return math.sqrt((sum1*sum1)+(w*self.func(violations))*(w*self.func(violations)))

    def func(self,sum1):
        return 1 + math.sqrt(sum1)*math.log(sum1/2,math.e)

    def getDist(self,S,team,currPlace,nextPlace):
        currPlace = -currPlace
        nextPlace = -nextPlace
        if currPlace < 0:
            currPlace = team
        if nextPlace < 0:
            nextPlace = team
        return self.distanceMap[currPlace-1,nextPlace-1]

    def getViolations(self,S):
        violations = 0
        team = 0
        roundN = 0
        count = 0
        while 1:
            if roundN > 2*self.n - 3:
                if abs(count) > 3:
                    violations = violations + 1
                roundN = 0
                team = team + 1
                count = 0
            if team == self.n:
                break

            if roundN == 0:
                count = np.sign(S[team,roundN])
            else:
                if np.sign(S[team,roundN])*np.sign(S[team,(roundN-1)]) == -1:
                    if abs(count) > 3:
                        violations = violations + 1
                    count = 0
                if np.sign(S[team,roundN]) == 1:
                    count = count + 1
                else:
                    count = count - 1

                if abs(S[team,roundN]) == abs(S[team,roundN-1]):
                    violations = violations + 1

            roundN = roundN + 1

        return violations

    def simulatedAnnealing(self):
        bestFeasible = np.Inf
        nbf = np.Inf
        bestInfeasible = np.Inf
        nbi = np.Inf
        reheat = 0
        counter = 0

        maxR = config.maxR
        maxP = config.maxP
        maxC = config.maxC
        T = config.T
        theta = config.theta
        sigma = config.sigma
        beta = config.beta
        w = config.w

        summaryFile = '''
        Run
        maxR = {}
        maxP = {}
        maxC = {}
        T = {}
        theta = {}
        beta = {}
        sigma = {}
        w = {}
        
        Input Solution - >
        
        {}
        
        Inital Cost - >
        
        {}
        '''.format(maxR,maxP,maxC,T,theta,beta,sigma,w,self.scheduleMap,self.cost((self.scheduleMap)))


        while reheat <= maxR:
            phase = 0
            while phase <= maxP:
                counter = 0
                while counter <= maxC:
                    #select a random move
                    S,St = self.randomMove()
                    costS = self.cost(S)
                    costSt = self.cost(St)
                    violationsS = self.getViolations(S)
                    violationsSt = self.getViolations(St)
                    if costSt < costS or violationsSt == 0 or costSt < bestFeasible or costS > 0 and violationsS < bestInfeasible:
                        accept = True
                    else:
                        if math.exp(-abs(costS - costSt) / T) > random.random():
                            accept = True
                        else:
                            accept = False
                    if accept:
                        S = St
                        if self.getViolations(S) == 0:
                            nbf = min(self.cost(S), bestFeasible)
                        else:
                            nbi = min(self.cost(S), bestInfeasible)
                        if nbf < bestFeasible or nbi < bestInfeasible:
                            reheat = 0
                            counter = 0
                            bestTemperature = T
                            bestFeasible = nbf
                            bestInfeasible = nbi
                            if self.getViolations(S):
                                w = w/theta
                            else:
                                w = w*sigma
                        else:
                            counter = counter + 1
                phase = phase + 1
                T = T*beta
            reheat = reheat + 1
            T = 2*bestTemperature
        summaryFile = summaryFile + '''
        
        Final Solution - >
        
        {}
        
        Final Cost - >
        
        {}
        '''.format(self.scheduleMap,self.cost(self.scheduleMap))
        writer = open('''{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'''.format(maxR,maxP,maxC,T,theta,beta,sigma,w,self.n),'w+')
        writer.write(summaryFile)
        writer.close()
    def randomMove(self):
        choice = random.randint(0,3)
        print(choice)
        if choice == 0:
            S = np.copy(self.scheduleMap)
            randTeamA,randTeamB = random.sample(range(1,self.n+1),2)
            self.swapTeams(randTeamA,randTeamB)
            St = np.copy(self.scheduleMap)
        elif choice == 1:
            S = np.copy(self.scheduleMap)
            randTeamA, randTeamB = random.sample(range(1, self.n+1), 2)
            self.swapHomes(randTeamA, randTeamB)
            St = np.copy(self.scheduleMap)
        elif choice == 2:
            S = np.copy(self.scheduleMap)
            randRoundA,randRoundB = random.sample(range(1, 2*self.n - 1),2)
            self.swapRounds(randRoundA,randRoundB)
            St = np.copy(self.scheduleMap)
        elif choice == 3:
            S = np.copy(self.scheduleMap)
            randRoundA,randRoundB = random.sample(range(0, 2*self.n - 1), 2)
            randTeam = random.sample(range(1,self.n+1),1)[0]
            self.partialSwapRounds(randTeam,randRoundA,randRoundB)
            St = np.copy(self.scheduleMap)
        return S,St