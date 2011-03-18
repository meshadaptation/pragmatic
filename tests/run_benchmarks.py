#!/usr/bin/env python

import subprocess

def kmp(nthreads, type=None):
        env=""
        if type:
                if(nthreads>1):
                        env="KMP_AFFINITY=\"verbose,granularity=core,%s\""%type
        else:
                if nthreads==1:
                        return ""
                elif nthreads==2:
                        cores="0,6"
                elif nthreads==4:
                        cores="0,1,6,7"
                elif nthreads==6:
                        cores="0,1,2,6,7,8"
                elif nthreads==8:
                        cores="0,1,2,3,6,7,8,9"
                elif nthreads==10:
                        cores="0,1,2,3,4,6,7,8,9,10"
                elif nthreads==12:
                        cores="0,1,2,3,4,5,6,7,8,9,10,11"
                else:
                        print "ERROR: unexpected thread cnt, ", nthreads
                        return ""
                env="KMP_AFFINITY=\"verbose,granularity=core,proclist=["+cores+"],explicit\""

        return env

cx1_resources = {"harpers": (8, "select=1:ncpus=8:harpers=true"),
                 "nehalem": (8, "select=1:ncpus=8:nehalem=true"),
                 "westmere": (12, "select=1:ncpus=12:westmere=true"),
                 "nehalem_hyperthreads": (16, "select=1:ncpus=8:ompthreads=16:nehalem=true"),
                 "westmere_hyperthreads": (24, "select=1:ncpus=12:ompthreads=24:westmere=true")}

benchmarks = ("test_hessian_2d", "test_hessian_3d",
             "test_smooth_simple_2d", "test_smooth_simple_3d",
             "test_smooth_constrained_2d", "test_smooth_constrained_3d")


def qsub(benchmark, resources, nthreads, ntrials=5):
        basename=benchmark
        filename_pbs = basename+".pbs"
	file_pbs = open(filename_pbs, "w")
	file_pbs.write("""#PBS -N pragmatic
#PBS -l %s
#PBS -l place=scatter:excl

# Time required in hh:mm:ss
#PBS -l walltime=2:00:00

module load intel-suite/11.1
module load vtk

cd $PBS_O_WORKDIR/bin

log=../%s.log
likwid-topology -g &> $log
ps -eo user,psr,pcpu,comm --sort pcpu | tail >> $log

for((n=0;n<%d;n++))
do
"""%(resources, basename, ntrials))
	for nthread in nthreads:
		file_pbs.write("echo \"Running trial $n on %d threads.\" >> $log\n"%nthread)
                file_pbs.write("OMP_NUM_THREADS=%d ./%s >> $log 2>&1\n"%(nthread, benchmark))
                file_pbs.write("%s OMP_NUM_THREADS=%d ./%s >> $log 2>&1\n"%(kmp(nthread), nthread, benchmark))
                file_pbs.write("%s OMP_NUM_THREADS=%d ./%s >> $log 2>&1\n"%(kmp(nthread, type="scatter"), nthread, benchmark))
        file_pbs.write("""
    done
done
""")
        file_pbs.close()
        # subprocess.Popen(["qsub", "-q", "pqcvcsm", filename_pbs])

for benchmark in benchmarks[5],:
	qsub(benchmark=benchmark, resources=cx1_resources["westmere"][1], nthreads=[1,2,4,6,8,10,12])
