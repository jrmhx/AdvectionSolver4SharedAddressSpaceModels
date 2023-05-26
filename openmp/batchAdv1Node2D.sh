#!/bin/bash
#PBS -q express
#PBS -j oe
#PBS -l walltime=00:01:00,mem=32GB
#PBS -l wd
#PBS -l ncpus=48
#

e= #echo

r=10
M=1000 # may need to be bigger
N=$M

numactl1="numactl --cpunodebind=0 --membind=0"

echo OMP_NUM_THREADS=1 $numactl1 ./testAdvect $M $N $r
OMP_NUM_THREADS=1 $e $numactl1 ./testAdvect $M $N $r
echo ""

ps="8 12 24 48"

for p in $ps; do
    opts="-P 4"
    echo ""
    if [ $p -le 24 ] ; then
	numactl="numactl --cpunodebind=0 --membind=0"
    else
	numactl=
    fi
    echo OMP_NUM_THREADS=$p $numactl ./testAdvect $opts $M $N $r
    OMP_NUM_THREADS=$p $e $numactl ./testAdvect $opts $M $N $r
    echo ""
done


exit

# OMP_NUM_THREADS=9 ./testAdvect -P 3 -x 1000 1000 100
