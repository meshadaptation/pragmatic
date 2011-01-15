#!/usr/bin/env python

cpus = (("harpers", 8, "ncpus=8:harpers=true"),
        ("nehalem", 8, "ncpus=8:nehalem=true"),
        ("nehalem_hyperthreads", 16, "ncpus=8:ompthreads=16:nehalem=true"),
        ("westmere", 12, "ncpus=12:westmere=true"),
        ("westmere_hyperthreads", 24, "ncpus=12:ompthreads=24:westmere=true"))

for cpu in cpus:
	file_pbs = open("%s.pbs"%cpu[0], "w")
	file_pbs.write("#PBS -N %s"%cpu[0][0:min(len(cpu[0]),12)])
        file_pbs.write("""
#PBS -l select=1:%s
#PBS -l place=scatter:excl

# Time required in hh:mm:ss
#PBS -l walltime=1:00:00

module load intel-suite/11.1
module load vtk

cd $PBS_O_WORKDIR

log=%s.log
cat /dev/null > $log
./benchmark.sh ./test_smooth_3d box20x20x20.vtu 5 %d >> $log 2>&1

cat /proc/cpuinfo >> $log
"""%(cpu[2], cpu[0], cpu[1]))
        file_pbs.close()


