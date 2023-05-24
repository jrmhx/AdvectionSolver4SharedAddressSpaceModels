#!/bin/bash

# Set the problem size (M and N) and radius (r)
M=4096
N=4096
r=1

# Define the range of values for Gx, Gy, Bx, and By
Gx_values=(1 2 4 8 16 32 64 128)
Gy_values=(1 2 4 8 16 32 64 128)
Bx_values=(1 2 4 8 16 32 64 128)
By_values=(1 2 4 8 16 32 64 128)

# Loop over different combinations of Gx, Gy, Bx, and By
for Gx in "${Gx_values[@]}"; do
  for Gy in "${Gy_values[@]}"; do
    for Bx in "${Bx_values[@]}"; do
      for By in "${By_values[@]}"; do
        # Run the testAdvect program with the current combination
        if ((Gx * Gy * Bx * By == 1024)); then
          # Run the testAdvect program with the current combination
          ./testAdvect -g $Gx,$Gy -b $Bx,$By -d 3 $M $N $r

          # Add any additional commands or processing you need here
        fi
        # Add any additional commands or processing you need here

      done
    done
  done
done
