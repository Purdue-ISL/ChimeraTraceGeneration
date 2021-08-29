#Chimera Trace Generation
#Described and used in the paper: 
#R. Shirey, S. Rao, and S. Sundaram, “Chimera: exploiting UAS flight path information to optimize heterogeneous data transmission,” in 2021 IEEE 29th International Conference on Network Protocols (ICNP). IEEE, 2021.
#Please cite our paper if you use this code
#Code by Russell Shirey, Purdue University

#MIT License

#Copyright (c) 2021 Russell Shirey

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import random
from scipy.stats import cauchy
import numpy as np

traceCount = 10 # number of traces to generate
orientationChange = 623 # point where flight changes orientation (varies based on flight path)
flightPathFilename = 'chimera_distances'

def initializeSingleDistances():
    f = open(flightPathFilename, 'r') # Flight path file
    rawDist = f.readlines()
    f.close()
    intervalDist = [float(i) for i in rawDist] # 1 second distance intervals
    return intervalDist

for iteration in range(1,traceCount+1):  # create 10 traces at a time (can be modified by changing the value of traceCount)
    bwArray = []
    random.seed(iteration-1) # each iteration will have different random seeds
    myDistances = initializeSingleDistances()
    flightLength = len(myDistances)
    
    f = open("trace"+str(iteration), "w")
    # make first throughput random between 14 and 16, based on actual data (users can experiment with higher and lower initial values)
    randomInt = random.randint(0, 100)
    prevThroughput = 14.00 + randomInt*1.0/50.0
    bwArray.append(prevThroughput)
    nextPrediction = 0.0

    # calculate error distributions (truncate with lower and upper limit based on our paper, because Cauchy tail is too long)
    # experiment with different limits to generate different traces (we found the below values to fit our data well)
    lowerLimit = 0.05
    upperLimit = 0.95

    # pre-generate 100 samples from the Cauchy distribution for each bin by distance (2 mile) and orientation
    binGoingUnder2 = np.linspace(cauchy.ppf(lowerLimit, loc=0.11553710, scale=0.51287519),
                                 cauchy.ppf(upperLimit, loc=0.11553710, scale=0.51287519), 100)
    binGoing2to4 = np.linspace(cauchy.ppf(lowerLimit, loc=0.66560511, scale=0.69437924),
                               cauchy.ppf(upperLimit, loc=0.66560511, scale=0.69437924), 100)
    binGoing4to6 = np.linspace(cauchy.ppf(lowerLimit, loc=0.4299680, scale=1.0784650),
                               cauchy.ppf(upperLimit, loc=0.4299680, scale=1.0784650), 100)
    binGoingOver6 = np.linspace(cauchy.ppf(lowerLimit, loc=0.6790805, scale=0.8341151),
                                cauchy.ppf(upperLimit, loc=0.6790805, scale=0.8341151), 100)
    binComingUnder2 = np.linspace(cauchy.ppf(lowerLimit, loc=0.4368314, scale=0.8345500),
                                  cauchy.ppf(upperLimit, loc=0.4368314, scale=0.8345500), 100)
    binComing2to4 = np.linspace(cauchy.ppf(lowerLimit, loc=0.6925304, scale=1.1054587),
                                cauchy.ppf(upperLimit, loc=0.6925304, scale=1.1054587), 100)
    binComing4to6 = np.linspace(cauchy.ppf(lowerLimit, loc=0.4563116, scale=1.5054202),
                                cauchy.ppf(upperLimit, loc=0.4563116, scale=1.5054202), 100)
    binComingOver6 = np.linspace(cauchy.ppf(lowerLimit, loc=-0.3113359, scale=1.245),
                                 cauchy.ppf(upperLimit, loc=-0.3113359, scale=1.245), 100)

    # generate the flight loop throughput trace (length of distance)
    for i in range(0, flightLength):
        nextDist = myDistances[i]
        if(i < orientationChange):
            # going (use regression parameters based on real-world flight data)
            nextGoing = prevThroughput * 0.4167 - 0.7612 * nextDist + 9.38
            nextPrediction = nextGoing
        else:
            #coming (use regression parameters based on real-world flight data)
            nextComing = prevThroughput * 0.5336 - 0.8290 * nextDist + 7.5632
            nextPrediction = nextComing

        #randomly sample from the pre-generated Cauchy distribution samples for the appropriate bin (based on the flight path)
        randomCauchySelection = random.randint(0, 99)
        errorChange = 0

        if (i < orientationChange):
            if (myDistances[i] <= 2):
                errorChange = binGoingUnder2[randomCauchySelection]
            elif (myDistances[i] <= 4):
                errorChange = binGoing2to4[randomCauchySelection]
            elif (myDistances[i] <= 6):
                errorChange = binGoing4to6[randomCauchySelection]
            elif (myDistances[i] > 6):
                errorChange = binGoingOver6[randomCauchySelection]
        else:
            if (myDistances[i] <= 2):
                errorChange = binComingUnder2[randomCauchySelection]
            elif (myDistances[i] <= 4):
                errorChange = binComing2to4[randomCauchySelection]
            elif (myDistances[i] <= 6):
                errorChange = binComing4to6[randomCauchySelection]
            elif (myDistances[i] > 6):
                errorChange = binComingOver6[randomCauchySelection]

        # Use error 50% of the time (this visually fit our data better and ensures sufficient reliance on previous throughput).
        # This can be viewed as a mixture of distributions, where one distribution corresponds to a Cauchy distribution and the other is a degenerate random variable that is always 0.  
        # We experimented with different mixture models and found this choice to visually and statistically fit well. 
        useError = random.randint(0,99)
        if(useError >= 50):
            thisBW = nextPrediction + errorChange
        else:
            thisBW = nextPrediction

        # If generated throughput is negative, truncate to 0.
        if(thisBW <= 0):
            thisBW = 0
        f.write(str(thisBW) + "\n")
        bwArray.append(thisBW)
        prevThroughput = thisBW
    f.close()



