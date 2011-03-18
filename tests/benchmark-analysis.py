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
thread_cnt=[1, 2, 4, 6, 8, 10, 12]

opts, args = getopt.getopt(sys.argv[1:], 'p:t:b:')

for o, a in opts:
    if o == '-p':
      key, ocol = a.split(":")
      column = int(ocol)
    elif o == '-t':
        ntrials = int(a)
    elif o == '-b':
        benchmark = a

filename = args[0]
basename = filename[:-4]
file = open(filename, "r")

t = [[], [], []]
pos=0
for line in file:
    if line.count(key):
        t[pos] = t[pos] + [float(line.split()[column])]
        pos = (pos+1)%3

times = [array(t[0]), array(t[1]), array(t[2])]
nthreads = len(times[0])/ntrials

for i in range(3):
    times[i].shape = (ntrials, nthreads)

tmean = [array([times[0][:, i].mean() for i in range(nthreads)]),
         array([times[1][:, i].mean() for i in range(nthreads)]),
         array([times[2][:, i].mean() for i in range(nthreads)])]
         
tstd = [array([times[0][:, i].std() for i in range(nthreads)]),
        array([times[1][:, i].std() for i in range(nthreads)]),
        array([times[2][:, i].std() for i in range(nthreads)])]

terr = [tstd[0]/sqrt(ntrials),
        tstd[1]/sqrt(ntrials),
        tstd[2]/sqrt(ntrials)]

E = [array([tmean[0][0]/((i+1)*tmean[0][i]) for i in range(nthreads)]),
     array([tmean[1][0]/((i+1)*tmean[1][i]) for i in range(nthreads)]),
     array([tmean[2][0]/((i+1)*tmean[2][i]) for i in range(nthreads)])]
idealE = array([1 for i in range(nthreads)])

S = [array([tmean[0][0]/tmean[0][i] for i in range(nthreads)]),
     array([tmean[1][0]/tmean[1][i] for i in range(nthreads)]),
     array([tmean[2][0]/tmean[2][i] for i in range(nthreads)])]
idealS = array(thread_cnt)

print "Percentage errors: ", 100*terr[0]/tmean[0], 100*terr[1]/tmean[1], 100*terr[2]/tmean[2]

# Standard error for both efficiency and speedup
error = [array([(sqrt((tstd[0][i]/tmean[0][i])**2 + (tstd[0][0]/tmean[0][0])**2)/E[0][i])/sqrt(ntrials)
                for i in range(nthreads)]),
         array([(sqrt((tstd[1][i]/tmean[1][i])**2 + (tstd[1][0]/tmean[1][0])**2)/E[1][i])/sqrt(ntrials)
                for i in range(nthreads)]),
         array([(sqrt((tstd[2][i]/tmean[2][i])**2 + (tstd[2][0]/tmean[2][0])**2)/E[2][i])/sqrt(ntrials)
                for i in range(nthreads)])]

thread = array(thread_cnt)

print "tmean[0] = ", tmean[0], terr[0]
print "tmean[1] = ", tmean[1], terr[1]
print "tmean[2] = ", tmean[2], terr[2]

pylab.figure(0)
pylab.errorbar(thread, tmean[0], yerr=terr[0], label="default")
pylab.errorbar(thread, tmean[1], yerr=terr[1], label="custom")
pylab.errorbar(thread, tmean[2], yerr=terr[2], label="scatter")
pylab.legend(loc="best")
pylab.xlabel("Number of threads")
pylab.ylabel("Time (Seconds)")
pylab.savefig(basename+"-times.pdf")

pylab.figure(1)
pylab.errorbar(thread, E[0], yerr=error[0], label="default")
pylab.errorbar(thread, E[1], yerr=error[1], label="custom")
pylab.errorbar(thread, E[2], yerr=error[2], label="scatter")
pylab.legend(loc="best")
pylab.plot(thread, idealE, label="Ideal")
pylab.legend(loc=4)
pylab.xlabel("Number of threads")
pylab.ylabel("Parallel efficiency")
pylab.savefig(basename+"-parallel_efficiency.pdf")

pylab.figure(2)
pylab.errorbar(thread, S[0], yerr=error[0], label="default")
pylab.errorbar(thread, S[1], yerr=error[1], label="custom")
pylab.errorbar(thread, S[2], yerr=error[2], label="scatter")
pylab.legend(loc="best")
pylab.plot(thread, idealS, label="Ideal")
pylab.xlabel("Number of threads")
pylab.ylabel("Speedup")
pylab.savefig(basename+"-speedup.pdf")
pylab.clf()
