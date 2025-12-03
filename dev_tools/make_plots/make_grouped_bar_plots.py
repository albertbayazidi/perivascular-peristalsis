import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


base_data_path = os.path.join("results", "comparison", "experiments", "bojorskaite_results")
ts_per_cycle = 25

sns.set_theme(style="whitegrid", context="talk", palette="Set2")
structures = {
    "single_element": f"ts_per_cycle_{ts_per_cycle}_single_element",
    "tandem": f"ts_per_cycle_{ts_per_cycle}_tandem",
    "bifurcated": f"ts_per_cycle_{ts_per_cycle}_bifurcated",
    "arterial_tree_n_3": f"ts_per_cycle_{ts_per_cycle}_n_3",
}

records = []

for struct, folder in structures.items():
    struct_path = os.path.join(base_data_path, folder)
    if not os.path.exists(struct_path):
        continue

    for exp in os.listdir(struct_path):
        exp_path = os.path.join(struct_path, exp)
        file = os.path.join(exp_path, "results.json")
        if not os.path.exists(file):
            continue

        with open(file) as f:
            d = json.load(f)

        r0 = d.get("r0")
        state = "REM" if r0 == 0.07 else "Non-REM"
        L = d.get("Ls", [None])[0]

        s = d.get("sweep", {})
        for t, f_val, lam, qn, qa in zip(
            s.get("ts_per_cycle", []),
            s.get("freq", []),
            s.get("lambdas", []),
            s.get("Q_avg_num", []),
            s.get("Q_avg_analytical", []),
        ):
            records += [
                dict(
                    Experiment=exp,
                    Structure=struct,
                    Sleep=state,
                    L=L,
                    ts_per_cycle=t,
                    freq=f_val,
                    lambda_=lam,
                    Type="Numerical",
                    FlowRate=float(qn),
                ),
                dict(
                    Experiment=exp,
                    Structure=struct,
                    Sleep=state,
                    L=L,
                    ts_per_cycle=t,
                    freq=f_val,
                    lambda_=lam,
                    Type="Analytical",
                    FlowRate=float(qa),
                ),
            ]

df = pd.DataFrame(records)

output_dir = os.path.join("results", "tables", f"ts_per_cycle_{ts_per_cycle}")
os.makedirs(output_dir, exist_ok=True)

for (L_val, sleep_state), df_group in df.groupby(["L", "Sleep"]):
    # Inside each plot, we create a grid using 'row' and 'col'
    # to show the 9 experiments (3 lambdas x 3 freqs).
    g = sns.catplot(
        data=df_group,
        x="Structure",
        y="FlowRate",
        hue="Type",  # "Numerical" vs "Analytical"
        col="freq",  # Creates 3 columns for the 3 frequencies
        row="lambda_",  # Creates 3 rows for the 3 lambdas
        kind="bar",
        palette="Set2",
        errorbar=None,
        height=4,  # Adjust height
        aspect=0.8,  # Adjust aspect ratio
    )

    g.set_axis_labels("Arterial Structure", "Flow Rate (Q_avg)")

    g.set_titles(r"$\lambda$={row_name} | $f$={col_name}")

    g.figure.suptitle(f"L={L_val}, {sleep_state} Sleep", fontsize=16, y=1.05)

    sns.move_legend(g, "upper left", bbox_to_anchor=(1.05, 1))

    g.set_xticklabels(rotation=45, ha="right")
    g.tight_layout()

    filename = os.path.join(output_dir, f"FlowRate_L{L_val}_{sleep_state}.png")

    g.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(g.figure)

print(f"Successfully generated 6 plot files in {output_dir}")
