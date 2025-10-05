#!/bin/bash
# experiments from paper
eps="--eps 0.1"
r0="--radius0 0.1"
depth="--depth 2"
lambdas="--lambdas 0.1 1 10 100"
ts_per_cycle="--ts_per_cycle 25 50"
freq="--freq 1"

# SINGLE PERIVASCULAR ELEMENT
## Freq experiments
python -m simulation.new_comparison $eps $r0 $depth $ts_per_cycle --betas 2 --Ls 1 --lambdas 1 --freq 0.1 1 10 100 &

## Lambda experiments
python -m simulation.new_comparison $eps $r0 $depth $lambdas $freq --betas 2 --Ls 1 --ts_per_cycle 50 100 &

# TANDEM PERIVASCULAR ELEMENTS
## Equal length segments
python -m simulation.new_comparison $eps $r0  $depth $lambdas $ts_per_cycle $freq --betas 2 3 --Ls 0.5 0.5  &

## Unequal length segments
python -m simulation.new_comparison $eps $r0 $depth $lambdas $ts_per_cycle $freq --betas 4 2 --Ls 1 2 & 

## BIFURCATION
## Equal children
python -m simulation.new_comparison $eps $r0 $depth $lambdas $ts_per_cycle $freq --betas 3 2 2 --Ls 2 1 1 &

## Unequal children
python -m simulation.new_comparison $eps $r0 $depth $lambdas $ts_per_cycle $freq --betas 2 3 4 --Ls 1 2 3 &

wait
