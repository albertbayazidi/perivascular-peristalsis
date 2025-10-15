build the apptainer image with 
```bash
sudo apptainer build graphnics.sif graphnics.def
```

Run command within apptainer like this
```bash
apptainer exec --bind /home/albert:/home/fenics/shared graphnics.sif \
python -m simulation.new_comparison \
--depth 7 --betas 2 --Ls 1 --lambdas 1 --freq 1 --ts_per_cycle 10
```

it can also be ran interactively 



