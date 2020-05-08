#!/bin/sh

# for evaluation, vary i from 1 to 1
# for control, vary i from 1 to 15

for ((i = 1; i <= 1; i++)); do
    python examples.py --seed 1 --algo mrpvf --decomp_type full_eig --num_eigvals 80 --transfer '' --run_index $i --stochastic '' --control True
done
