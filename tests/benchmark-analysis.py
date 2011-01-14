#!/usr/bin/env python

from math import sqrt
from numpy import array
import pylab
import sys

# default data source
file = sys.stdin
column = 0
ntrials = 1

if len(sys.argv) == 2:
    ntrials = int(sys.argv[1])
if len(sys.argv) == 3:
    ntrials = int(sys.argv[1])
    column = int(sys.argv[2])
elif len(sys.argv) == 4:
    file = open(sys.argv[1], "r")
    ntrials = int(sys.argv[2])
    column = int(sys.argv[3])

times = array([float(line.split()[column]) for line in file])
nthreads = len(times)/ntrials

times.shape = (ntrials, nthreads)

tmean = array([times[:, i].mean() for i in range(nthreads)])
tstd = array([times[:, i].std() for i in range(nthreads)])

E = array([tmean[0]/((i+1)*tmean[i]) for i in range(nthreads)])
idealE = array([1 for i in range(nthreads)])

S = array([tmean[0]/tmean[i] for i in range(nthreads)])
idealS = array([i+1 for i in range(nthreads)])

# Standard error for both efficiency and speedup
error = array([(sqrt((tstd[i]/tmean[i])**2 + (tstd[0]/tmean[0])**2)/E[i])/sqrt(ntrials) for i in range(nthreads)])

thread = array([i+1 for i in range(nthreads)])

#pylab.plot(thread, E)
#pylab.plot(thread, idealE)
#pylab.show()

pylab.plot(thread, S)
pylab.plot(thread, idealS)
pylab.show()
