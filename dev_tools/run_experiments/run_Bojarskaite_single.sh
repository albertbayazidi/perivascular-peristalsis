depth="--depth 1"
ts_per_cycle="--ts_per_cycle 50"
n_cycle="--n_cycle 50"

wave_speed=0.4
lambda="--lambdas $(awk "BEGIN {printf \"%.3f\", $wave_speed/0.3}")"
freq="--freq 0.1 0.2 0.3" 

NONREM_ARGS="--radius0 0.006 --betas 2.167 --eps 0.167"
REM_ARGS="--radius0 0.0075 --betas 1.8 --eps 0.067"

# non REM sleep
python -m simulation.new_comparison $depth $ts_per_cycle $n_cycle $lambda $freq $NONREM_ARGS --Ls 0.637 &
python -m simulation.new_comparison $depth $ts_per_cycle $n_cycle $lambda $freq $NONREM_ARGS --Ls 0.318 &
python -m simulation.new_comparison $depth $ts_per_cycle $n_cycle $lambda $freq $NONREM_ARGS --Ls 0.212 &

