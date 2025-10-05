#!/bin/bash
depth="--depth 2"

# could use these but i want to test computing them based on c/f = lambda
wave_speed_mesentry="--lambdas 0.1" # Daversin-Catty 2020 se tabl II in directional flow
wave_speed_pial="--lambdas 0.4" 

# Data from Bojarskaite article
eps=""
r0="--radius0 6"
betas="--betas 2.16"

ts_per_cycle="--ts_per_cycle 10 "  # should mabye be ran for longer

vasomotion_freq="--freq 0.1 0.2 0.3"
respiratory_freq="--freq 0.3 0.6 0.9"

# WAKE

# NONREM


# REM


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

