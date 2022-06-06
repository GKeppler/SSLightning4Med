"""
This script downlaods metrics from Wandb, plots them, produces a latex table and saves results in a CSV
"""
#%%
import os
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb

#%%
# Project is specified by <entity/project-name>
project_name = "test2"
# most important filters are: "display_name", "sweep" in combination with or:
# e.g "$or": [{"display_name":"twilight-dew-182"},{"sweep": "bqi2hy0k"}, {"sweep": "8827ug5t"}]
sweep_id = "bqi2hy0k"
filters = {
    # "$or": [{"state": "finished"}, {"state": "crashed"}],
    "state": "finished",
    "$or": [{"display_name": "twilight-dew-182"}, {"sweep": "bqi2hy0k"}, {"sweep": "8827ug5t"}],  #
    # "sweep": sweep_id,
}
# download data
api = wandb.Api()
runs = api.runs(f"gkeppler/{project_name}", filters=filters)
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
print(len(all_df))
all_df.head()
#%%
# preprocessing
all_df["split"] = all_df["split"].apply(lambda x: x.replace("_", "/"))

# check if multiple dataset are in the sweep
if len(all_df["dataset"].unique()) > 1:
    print("more that on dataset in the metrics", all_df["dataset"].unique())

# drop rows with na in column "test mDICE"
all_df = all_df.dropna(subset=["test mDICE"])
all_df.head()
# %%
# calculate metrics
# for a given methods this will calculate the mean and std of the metrics

# specify column names for metrics that are plotted later
# ONLY WHEN A SUPERVISED ONLY TEST RUN WAS PERFORMED IS multiple_tests = True
multiple_tests = False
if multiple_tests:
    # get history for first iteration of test sample
    runs = api.runs(f"gkeppler/{project_name}")
    metrics = ["test mIOU", "test mDICE", "test overall_acc"]
    list_metrics = []
    for i, run in enumerate(runs):
        for j, row in run.history(keys=metrics).iterrows():
            # get only first elements of history -> supervised metric
            row.name = i
            list_metrics.append(row)
            break

    # combine with additional metrics
    sup_df = pd.DataFrame(list_metrics).add_suffix(" sup")
    sup_df.head()
    all_df = pd.concat([all_df, sup_df], axis=1)

    columns = [
        "split",
        "test overall_acc",
        "test overall_acc sup",
        "test mIOU",
        "test mIOU sup",
        "test mDICE",
        "test mDICE sup",
    ]
else:
    columns = [
        "split",
        "test overall_acc",
        "test mIOU",
        "test mDICE",
    ]

metrics_df = all_df[columns].groupby("split").agg([np.mean, np.std, np.count_nonzero])
metrics_df = metrics_df.reindex(["1", "1/4", "1/8", "1/30"])
metrics_df.head()
# drop Na
metrics_df.dropna(axis=0, how="all", inplace=True)
metrics_df.head()
# columns are nested with mean, std in second row -> unzip and get list
columns_table = list(set([el[0] for el in metrics_df.columns.tolist()]))
# generate table
# function to generate text from merics to use in latex table
def make_text(s: Dict) -> str:
    mean = s["mean"]
    std = s["std"]
    return f"{mean*100:.2f}({std*100:.2f})"


# generate latex tables
text_df = pd.DataFrame()
for col in columns_table:
    text_df[col] = metrics_df[col].apply(make_text, axis=1)
print(text_df.transpose().to_latex())
#%%
# plot figures
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

fig, ax = plt.subplots()
for col in columns_table:
    metrics_df[col].reset_index().plot("split", "mean", yerr="std", label=col[5:], ax=ax)
plt.savefig(f"{project_name}_metrics.pdf", transparent=False, bbox_inches="tight")
#%%
# methods comparison

metric_dict = {
    "mean Average Surface Distance": "test medpy_asd",
    "mean Hausdorff Distance": "test medpy_hd",
    "mean DSC": "test medpy_dc",
    "mean IoU": "test medpy_jc",
    "mean pixel accuracy": "test overall_acc",
}  # "test mDICE"#"test mIOU"

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
for metric_name in metric_dict.keys():
    metric = metric_dict[metric_name]
    columns = [
        "split",
        "method",
        metric,
    ]
    method_df = all_df[columns].groupby(["method", "split"]).agg([np.mean, np.std, np.count_nonzero])
    method_df.dropna(axis=0, how="all", inplace=True)
    columns_table = list(set([el[0] for el in method_df.columns.tolist()]))
    method_df.head()

    # draw figure
    fig, ax = plt.subplots()

    for label, group in method_df.reset_index().groupby("method"):
        # add a stepped horizontal lien to the plot
        if label == "Supervised":
            # get value of metric where split is 1
            value = group.loc[group["split"] == "1", metric]["mean"].values[0]
            print(value)
            # drop this row from the dataframe "group"
            group = group.drop(group[group["split"] == "1"].index)
            ax.axhline(y=value, color="k", linestyle="--")
        group.columns = ["".join(col) for col in group.columns]
        # replace Na with 0
        group[metric + "std"].replace(np.NaN, 0, inplace=True)
        group["order"] = group["split"].str.split("/").apply(lambda x: float(x[0]) / float(x[1]) if len(x) > 1 else 1)
        print(group.head())
        group.sort_values(by="order").plot("split", metric + "mean", yerr=metric + "std", label=label, ax=ax)
    ax.set_title("Metric: " + metric_name + "\n" + "Dataset: " + all_df["dataset"].iloc[0])
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig(f"figures/{project_name}_{sweep_id}_{metric}_methods.pdf", transparent=False, bbox_inches="tight")

# %%
