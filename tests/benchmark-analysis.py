#!/usr/bin/env python

from math import sqrt
from numpy import array
import pylab
import sys
import getopt

# Defaults
files = []
column = 0
ntrials = 1
key=""

opts, args = getopt.getopt(sys.argv[1:], 'p:t:b:')

for o, a in opts:
    if o == '-p':
        key, ocol = a.split(":")
        column = int(ocol)
    elif o == '-t':
        ntrials = int(a)
    elif o == '-b':
        benchmark = a

if len(args)==0:
    files = [sys.stdin]
else:
    for file in args:
        files = files + [(open(file, "r"), file[:-4])]

for file, filename in files:
    t = []
    for line in file:
        if line.count(key):
            t = t + [float(line.split()[column])]
    times = array(t)
    nthreads = len(times)/ntrials

    times.shape = (ntrials, nthreads)

    tmean = array([times[:, i].mean() for i in range(nthreads)])
    tstd = array([times[:, i].std() for i in range(nthreads)])
    terr = tstd/sqrt(ntrials)

    E = array([tmean[0]/((i+1)*tmean[i]) for i in range(nthreads)])
    idealE = array([1 for i in range(nthreads)])

    S = array([tmean[0]/tmean[i] for i in range(nthreads)])
    idealS = array([i+1 for i in range(nthreads)])

    print "Percentage errors: ", 100*terr/tmean

    # Standard error for both efficiency and speedup
    error = array([(sqrt((tstd[i]/tmean[i])**2 + (tstd[0]/tmean[0])**2)/E[i])/sqrt(ntrials) for i in range(nthreads)])

    thread = array([i+1 for i in range(nthreads)])

    pylab.figure(0)
    pylab.errorbar(thread, tmean, yerr=terr, label=filename)

    pylab.figure(1)
    pylab.errorbar(thread, E, yerr=error, label=filename)
    
    pylab.figure(2)
    pylab.errorbar(thread, S, yerr=error, label=filename)

pylab.figure(0)
pylab.legend(loc="best")
pylab.xlabel("Number of threads")
pylab.ylabel("Time (Seconds)")
pylab.savefig("times.pdf")

pylab.figure(1)
pylab.plot(thread, idealE, label="Ideal")
pylab.legend(loc="best")
pylab.xlabel("Number of threads")
pylab.ylabel("Parallel efficiency")
pylab.savefig("parallel_efficiency.pdf")

pylab.figure(2)
pylab.plot(thread, idealS, label="Ideal")
pylab.legend(loc="best")
pylab.xlabel("Number of threads")
pylab.ylabel("Speedup")
pylab.savefig("speedup.pdf")
pylab.clf()

