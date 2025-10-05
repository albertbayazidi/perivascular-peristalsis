To run a program use 

```bash
python -m filename #
```

The comparison between the analytical and numerical approach on a single vessel can be run by executing
```bash
python  -m simulation.run_comparison --betas 2 --Ls 1 --ts_per_cycle 25 --lambdas 0.1 1 2 10 
```

The number of arguments given to --betas and --Ls determines the number of vessel segments; running e.g.
```bash
python -m simulation.run_comparison --betas 2 2 2 --Ls 1 1 1 --ts_per_cycle 25 --lambdas 0.1 1 2 10 
```

The simulation of interacting cardiac and vasomotion waves can be run by executing

```bash
python -m simulation.run_interaction_simulation
```

