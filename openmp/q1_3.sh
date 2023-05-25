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

numactl1="numactl --cpunodebind=0 --membind=0"


ps="4 8 16 32 48"
p_s="1 2 4 8"

echo "1D"
echo ""
# 1D
for p in $ps; do
    opts=""
    echo ""
    echo OMP_NUM_THREADS=$p ./testAdvect $opts $M $N $r
    OMP_NUM_THREADS=$p $e ./testAdvect $opts $M $N $r
    echo ""
done

echo "2D_1"
echo ""
# 2D_1
for p in $ps; do
    opts="-P 4"
    echo ""
    echo OMP_NUM_THREADS=$p ./testAdvect $opts $M $N $r
    OMP_NUM_THREADS=$p $e ./testAdvect $opts $M $N $r
    echo ""
done

echo "2D_2"
echo ""
# 2D_2
for p in $p_s; do
    opts=""
    echo ""
    echo OMP_NUM_THREADS=48 $numactl ./testAdvect -P $p $M $N $r
    OMP_NUM_THREADS=48 $e $numactl ./testAdvect -P $p $M $N $r
    echo ""
done

# OMP_NUM_THREADS=9 ./testAdvect -P 3 -x 1000 1000 100
