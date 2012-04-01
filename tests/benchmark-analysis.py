#!/usr/bin/env python

from math import sqrt
from numpy import array
import pylab
import sys
import getopt

# Defaults
ntrials = 10
key=""
thread_cnt=[1, 2, 4, 6, 8, 10, 12]

opts, args = getopt.getopt(sys.argv[1:], 'p:')

for o, a in opts:
    if o == '-p':
      key = a

filename = args[0]
basename = filename[:-4]
file = open(filename, "r")

t = [[], [], []]
pos=0
for line in file:
    if line.count(key):
        t[pos] = t[pos] + [float(line.split()[-1])]
        pos = (pos+1)%3

times = [array(t[0]), array(t[1]), array(t[2])]
nthreads = len(times[0])/ntrials

for i in range(3):
    times[i].shape = (ntrials, nthreads)

tmean = [array([times[0][:, i].mean() for i in range(nthreads)]),
         array([times[1][:, i].mean() for i in range(nthreads)]),
         array([times[2][:, i].mean() for i in range(nthreads)])]
         
terr = [array([times[0][:, i].std()/sqrt(ntrials) for i in range(nthreads)]),
        array([times[1][:, i].std()/sqrt(ntrials) for i in range(nthreads)]),
        array([times[2][:, i].std()/sqrt(ntrials) for i in range(nthreads)])]

E = [array([(times[0][:, 0]/(thread_cnt[i]*times[0][:, i])).mean() for i in range(nthreads)]),
     array([(times[1][:, 0]/(thread_cnt[i]*times[1][:, i])).mean() for i in range(nthreads)]),
     array([(times[2][:, 0]/(thread_cnt[i]*times[2][:, i])).mean() for i in range(nthreads)])]

Eerr = [array([(times[0][:, 0]/(thread_cnt[i]*times[0][:, i])).std()/sqrt(ntrials) for i in range(nthreads)]),
        array([(times[1][:, 0]/(thread_cnt[i]*times[1][:, i])).std()/sqrt(ntrials) for i in range(nthreads)]),
        array([(times[2][:, 0]/(thread_cnt[i]*times[2][:, i])).std()/sqrt(ntrials) for i in range(nthreads)])]


S = [array([(times[0][:, 0]/times[0][:, i]).mean() for i in range(nthreads)]),
     array([(times[1][:, 0]/times[1][:, i]).mean() for i in range(nthreads)]),
     array([(times[2][:, 0]/times[2][:, i]).mean() for i in range(nthreads)])]

Serr = [array([(times[0][:, 0]/times[0][:, i]).std()/sqrt(ntrials) for i in range(nthreads)]),
        array([(times[1][:, 0]/times[1][:, i]).std()/sqrt(ntrials) for i in range(nthreads)]),
        array([(times[2][:, 0]/times[2][:, i]).std()/sqrt(ntrials) for i in range(nthreads)])]

idealS = array(thread_cnt)
thread = array(thread_cnt)

pylab.figure(0)
pylab.errorbar(thread, tmean[0], yerr=terr[0], label="default")
pylab.errorbar(thread, tmean[1], yerr=terr[1], label="custom")
pylab.errorbar(thread, tmean[2], yerr=terr[2], label="scatter")
pylab.legend(loc="best")
pylab.xlabel("Number of threads")
pylab.ylabel("Time (Seconds)")
pylab.savefig(basename+"-times.pdf")

pylab.figure(1)
pylab.errorbar(thread, E[0], yerr=Eerr[0], label="default")
pylab.errorbar(thread, E[1], yerr=Eerr[1], label="custom")
pylab.errorbar(thread, E[2], yerr=Eerr[2], label="scatter")
pylab.legend(loc="best")
pylab.xlabel("Number of threads")
pylab.ylabel("Parallel efficiency")
pylab.savefig(basename+"-parallel_efficiency.pdf")

pylab.figure(2)
pylab.errorbar(thread, S[0], yerr=Serr[0], label="default")
pylab.errorbar(thread, S[1], yerr=Serr[1], label="custom")
pylab.errorbar(thread, S[2], yerr=Serr[2], label="scatter")
pylab.legend(loc="best")
pylab.plot(thread, idealS, label="Ideal")
pylab.xlabel("Number of threads")
pylab.ylabel("Speedup")
pylab.savefig(basename+"-speedup.pdf")
pylab.clf()
