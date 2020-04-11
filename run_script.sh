#!/bin/sh

for ((i = 1; i <= 15; i++)); do
    python examples.py --seed 1 --algo mrpvf --decomp_type eig --num_eigvals 80 --transfer '' --run_index $i --stochastic True
done
