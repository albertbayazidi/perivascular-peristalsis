#!/bin/bash
# add & to end to have them run in "parallel", and wait at the end for safety

#lambda experiments
#python -m simulation.new_comparison --depth 2 --betas 2 --Ls 1 --lambdas 10 20 --freq 10 100 --ts_per_cycle 10 5
#python -m simulation.new_comparison --depth 2 --betas 2 --Ls 1 --lambdas 0.1 1 2 10 --freq 0.1 1 10 100 --ts_per_cycle 50

#freq experiments
#python -m simulation.new_comparison --depth 2 --betas 2 --Ls 1 --lambdas 1 --freq 0.1 1 10 100 --ts_per_cycle 50 
#python -m simulation.new_comparison --depth 2 --betas 2 --Ls 1 --lambdas 1 --freq 0.1 1 10 100 --ts_per_cycle 100 

# Tandem perivascular elements
#python -m simulation.new_comparison --depth 2 --betas 2 3 --Ls 0.5 0.5 --lambdas 0.1 1 2 10 --freq 1

#python -m simulation.new_comparison --depth 2 --betas 4 2 --Ls 1 2 --lambdas 0.1 1 2 10 --freq 1

# bifurcation
#python -m simulation.new_comparison --depth 2 --betas 3 2 2 --Ls 2 1 1 --lambdas 1 --freq 1 

