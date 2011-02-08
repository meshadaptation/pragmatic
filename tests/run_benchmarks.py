#!/usr/bin/env python

benchmark = "./test_smooth_simple_3d"
ntrials = 10

cpus = (("harpers", 8, "ncpus=8:harpers=true"),
        ("nehalem", 8, "ncpus=8:nehalem=true"),
        ("nehalem_hyperthreads", 16, "ncpus=8:ompthreads=16:nehalem=true"),
        ("westmere", 12, "ncpus=12:westmere=true"),
        ("westmere_hyperthreads", 24, "ncpus=12:ompthreads=24:westmere=true"))

for cpu in cpus:
        filename_pbs = "%s.pbs"%cpu[0]
	file_pbs = open(filename_pbs, "w")
	file_pbs.write("#PBS -N %s\n"%cpu[0][0:min(len(cpu[0]),12)])
	file_pbs.write("#PBS -l select=1:%s\n"%cpu[2])
        file_pbs.write("""
#PBS -l place=scatter:excl

# Time required in hh:mm:ss
#PBS -l walltime=1:00:00

module load intel-suite/11.1
module load vtk

cd $PBS_O_WORKDIR/bin

log=../%s.log
cat /proc/cpuinfo > $log

for((n=0;n<%d;n++))
do
    for((i=1;i<=%d;i++))
    do
        echo "###############################"
        echo "Running trial $n on $i threads."
        OMP_NUM_THREADS=$i %s
    done
done

"""%(cpu[0], ntrials, cpu[1], benchmark))
        file_pbs.close()
        qsub filename_pbs

