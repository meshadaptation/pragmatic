#!/usr/bin/env python

from math import sqrt
from numpy import array
import pylab
import sys
import getopt

# Defaults
file = sys.stdin
column = 0
ntrials = 1
benchmark="Benchmark"

opts, args = getopt.getopt(sys.argv[1:], 'c:t:b:')

for o, a in opts:
    if o == '-c':
        column = int(a)
    elif o == '-t':
        ntrials = int(a)
    elif o == '-b':
        benchmark = a

if len(args)>0:
    file = open(args[0], "r")

times = array([float(line.split()[column]) for line in file])
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

pylab.errorbar(thread, tmean, yerr=terr, label=benchmark)
pylab.legend(loc="best")
pylab.xlabel("Number of threads")
pylab.ylabel("Time (Seconds)")
pylab.savefig(benchmark+"_time.pdf")

pylab.plot(thread, idealE, label="Ideal")
pylab.errorbar(thread, E, yerr=error, label=benchmark)
pylab.legend(loc="best")
pylab.xlabel("Number of threads")
pylab.ylabel("Parallel efficiency")
pylab.savefig(benchmark+"_efficiency.pdf")

pylab.plot(thread, idealS, label="Ideal")
pylab.errorbar(thread, S, yerr=error, label=benchmark)
pylab.legend(loc="best")
pylab.xlabel("Number of threads")
pylab.ylabel("Speedup")
pylab.savefig(benchmark+"_speedup.pdf")
