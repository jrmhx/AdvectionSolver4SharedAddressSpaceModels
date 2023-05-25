#!/bin/bash
#PBS -q express
#PBS -j oe
#PBS -l walltime=00:01:00,mem=32GB
#PBS -l wd
#PBS -l ncpus=48
#

e= #echo

r=100
M=1000 # may need to be bigger
N=$M

p_s="1 3 6 8 12 24 48"

echo "2D_2"
echo ""
# 2D_2
for p in $p_s; do
    opts=""
    echo ""
    echo OMP_NUM_THREADS=48 ./testAdvect -P $p $M $N $r
    OMP_NUM_THREADS=48 $e ./testAdvect -P $p $M $N $r
    echo ""
done

# OMP_NUM_THREADS=9 ./testAdvect -P 3 -x 1000 1000 100
