depth="--depth 2"
ts_per_cycle="--ts_per_cycle 25"
n_cycle="25"

# har ingenting å basere valge av neste beta på
# valge neste L til å være samme lengde

NONREM_ARGS="--radius0 0.06 --betas 2.167 2.167 2.167 --eps 0.014 --lambdas 4.0 2.395 1.717 --freq 0.1 0.167 0.233"
REM_ARGS="--radius0 0.07 --betas 1.714 1.714 1.714 --eps 0.007 --lambdas 4.0 2.395 1.717 --freq 0.1 0.167 0.233 "

# non REM sleep
python -m simulation.new_comparison $depth $ts_per_cycle $NONREM_ARGS --Ls 6.283 6.283 6.283 &

python -m simulation.new_comparison $depth $ts_per_cycle $NONREM_ARGS --Ls 3.762 3.762 3.762 & 

python -m simulation.new_comparison $depth $ts_per_cycle $NONREM_ARGS --Ls 2.697 2.697 2.697 & 

# REM sleep
python -m simulation.new_comparison $depth $ts_per_cycle $REM_ARGS --Ls 6.283 6.283 6.283 &  

python -m simulation.new_comparison $depth $ts_per_cycle $REM_ARGS --Ls 3.762 3.762 3.762 & 

python -m simulation.new_comparison $depth $ts_per_cycle $REM_ARGS --Ls 2.697 2.697 2.697 &

