#!/bin/bash
ts_per_cycle="--ts_per_cycle 100"
n_cycles="--n_cycles 3"
depth="--depth 1"
status=0 

if [ $status -eq 0 ]; then
    echo "we in cardiac"
    #NON REM cardiac
    python -m simulation.main --radius0 0.006 --betas 2.167 --eps 0.027 $depth $ts_per_cycle $n_cycles --Ls 0.6 --lambdas 400 --freq 8 &

    #REM cardiac 
    python -m simulation.main --radius0 0.0075 --betas 1.8 --eps 0.017 $depth $ts_per_cycle $n_cycles --Ls 0.6 --lambdas 288 --freq 11.11 &

    wait
    path="./results/comparison/single_element/exp_e8c32134 ./results/comparison/single_element/exp_a50dd1c5"

    python -m dev_tools.make_plots.process_experiment_pairs $path --window 0 300 --nodes 0 1
elif [ $status -eq 1 ]; then
    echo "we in vasomotion"
    #NON REM vasomotion
    python -m simulation.main --radius0 0.006 --betas 2.167 --eps 0.167 $depth $ts_per_cycle $n_cycles --Ls 0.6 --lambdas 6.533 --freq 0.3 &

    #REM vasomotion 
    python -m simulation.main --radius0 0.0075 --betas 1.8 --eps 0.067 $depth $ts_per_cycle $n_cycles --Ls 0.6 --lambdas 6.533 --freq 0.3 &

    wait
    path="./results/comparison/single_element/exp_3725bd37 ./results/comparison/single_element/exp_2bcb6c1c"

    python -m dev_tools.make_plots.process_experiment_pairs $path --window 0 300 --nodes 0 1

else
    echo "we in vasomotion + cardiac"
    #NON REM vasomotion
    python -m simulation.main --radius0 0.006 --betas 2.167 --eps 0.167 $depth $ts_per_cycle $n_cycles --Ls 0.6 --lambdas 6.533 --freq 0.3 &

    #REM vasomotion 
    python -m simulation.main --radius0 0.0075 --betas 1.8 --eps 0.067 $depth $ts_per_cycle $n_cycles --Ls 0.6 --lambdas 6.533 --freq 0.3 &

    wait
    path="./results/comparison/single_element/exp_d85fb2aa ./results/comparison/single_element/exp_8850557f"

    python -m dev_tools.make_plots.process_experiment_pairs $path --window 0 300 --nodes 0 1
fi


