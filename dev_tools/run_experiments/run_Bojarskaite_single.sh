depth="--depth 1"
ts_per_cycle="--ts_per_cycle 25"
n_cycle="--n_cycle 5"

NONREM_ARGS="--radius0 0.06 --betas 2.167 --eps 0.167 --lambdas 4.0 --freq 0.1"
REM_ARGS="--radius0 0.075 --betas 1.8 --eps 0.067  --lambdas 4.0 --freq 0.1"
# non REM sleep
#python -m simulation.new_comparison $depth $ts_per_cycle $n_cycle $NONREM_ARGS --Ls 0.637 &

#python -m simulation.new_comparison $depth $ts_per_cycle $n_cycle $NONREM_ARGS --Ls 0.318 & 

#python -m simulation.new_comparison $depth $ts_per_cycle $n_cycle $NONREM_ARGS --Ls 0.212 & 

# REM sleep
#python -m simulation.new_comparison $depth $ts_per_cycle $n_cycle $REM_ARGS --Ls 0.637 &  

#python -m simulation.new_comparison $depth $ts_per_cycle $n_cycle $REM_ARGS --Ls 0.318 & 

#python -m simulation.new_comparison $depth $ts_per_cycle $n_cycle $REM_ARGS --Ls 0.212 &
