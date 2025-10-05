# Worth Remembering
we only compare and theirfore store the root flow, there might be a bigger diffrence between analytical flow when not looking at the root

If n_cyles is not provided then n_cyles will equal ts_per_cycle. 

we can provide mutiple ts_per_cycle however then we also need to do two things
- need to make make_dummy_data allow for more data * len(ts_per_cycle)
- need to store the changes in the config.json and that makes the file harder to read. 

# known issues and workaounds
- when running the run_experiments_from_main_aricle.sh one gets wired visual bugs in the console.
- Inside run_interaction_simulation, the class *TimeDepHydraulicNetwork* attempted to use 2nd-order polynomials for the Lagrange elements. This functionality did not actually exist. In the newest commit, *TimeDepHydraulicNetwork* no longer has support for changing the degree parameter. 
- Since only the first epxeriment is ran in run_interaction_simulation this means that plot_interaction_netflow should only look at the zero'th experiment. 

# Issues Not Jet Understood
Inside run_interaction_simulation there is a cell that is meant to plot something using plt, but this does not work. It tries to print outflows_root and outflows_leaf2, but these variables do not exist, and it is not very clear what they are. This needs to be investigated further.
