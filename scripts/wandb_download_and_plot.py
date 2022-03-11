"""
This script downlaods metrics from Wandb, plots them, produces a latex table and saves results in a CSV
"""

from typing import Dict

#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

#%%
# Project is specified by <entity/project-name>
project_name = "st++breastCancer"

# download data
api = wandb.Api()
runs = api.runs(f"gkeppler/{project_name}")
summary_list = []
config_list = []
name_list = []
for run in runs:
    # run.summary are the output key/values like accuracy.
    # We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # run.config is the input metrics.
    # We remove special values that start with _.
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    config_list.append(config)

    # run.name is the name of the run.
    name_list.append(run.name)


summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame({"name": name_list})
all_df = pd.concat([name_df, config_df, summary_df], axis=1)
all_df.head()

#%%
# get history for ferst iteration of test sample
runs = api.runs(f"gkeppler/{project_name}")
metrics = ["test mIOU", "test mDICE", "test overall_acc"]
list_metrics = []
for i, run in enumerate(runs):
    for j, row in run.history(keys=metrics).iterrows():
        # get only first elements of history -> supervised metric
        row.name = i
        list_metrics.append(row)
        break

#%%
sup_df = pd.DataFrame(list_metrics).add_suffix(" sup")
sup_df.head()
#%%
# preprocessing
# drop all failed runs
all_df = all_df.dropna(axis=0)
all_df["split"] = all_df["split"].apply(lambda x: x.replace("_", "/"))

# combine with additional metrics
all_df = pd.concat([all_df, sup_df], axis=1)

#%%
columns = [
    "split",
    "test overall_acc",
    "test overall_acc sup",
    "test mIOU",
    "test mIOU sup",
    "test mDICE",
    "test mDICE sup",
]

#%%
# calculate metrics
metrics_df = all_df[columns].groupby("split").agg([np.mean, np.std, np.count_nonzero])
metrics_df = metrics_df.reindex(["1", "1/4", "1/8", "1/30"])
metrics_df.head()
#%%
# columns are nested with mean, std in second row -> unzip and get list
columns = list(set([el[0] for el in metrics_df.columns.tolist()]))
# generate table
#%%
# function to generate text from merics to use in latex table


def make_text(s: Dict) -> str:
    mean = s["mean"]
    std = s["std"]
    return f"{mean*100:.2f}({std*100:.2f})"


#%%
# generate latex tables
text_df = pd.DataFrame()
for col in columns:
    text_df[col] = metrics_df[col].apply(make_text, axis=1)
print(text_df.transpose().to_latex())
#%%
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
#%%
# plot figures
fig, ax = plt.subplots()
for col in columns:
    metrics_df[col].reset_index().plot("split", "mean", yerr="std", label=col[5:], ax=ax)
# %%
# save plot and metrics
plt.savefig(f"{project_name}.pdf", transparent=False, bbox_inches="tight")
metrics_df.to_csv(f"{project_name}.csv")
# %%
