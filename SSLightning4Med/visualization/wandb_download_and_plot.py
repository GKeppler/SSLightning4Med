"""
This script downlaods metrics from Wandb, plots them, produces a latex table and saves results in a CSV
"""
#%%
import os
from typing import Dict

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d

import wandb

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

#%%
# Project is specified by <entity/project-name>
project_name = "SSLightning4Med"
# most important filters are: "display_name", "sweep" in combination with or:

# Zebrafish
name = "all train"
filters = {
    "Early-Stopping and best-checkpoint": {
        # "$or": [{"state": "finished"}, {"state": "crashed"}],
        "state": "finished",
        "$or": [{"sweep": "7m6repqe"}, {"sweep": "1g6zh54u"}, {"sweep": "8827ug5t"}],
        # "sweep": sweep_id,
    },
    "no early stopping": {"$or": [{"display_name": "twilight-dew-182"}, {"sweep": "bqi2hy0k"}, {"sweep": "8827ug5t"}]},
    "Supervised all": {
        "sweep": "s6s9lkos",
    },
    "all train": {
        "$or": [
            {"sweep": "9w5n97fs"},  # heureka CCT
            {"sweep": "t7yc3a60"},  # Supervised all correct St++ weak augmentations
            {"sweep": "tizubuhc"},  # Fixmatch, MeanTeacher, heureka
            {"sweep": "v24zcssb"},  # hippocampus, mean,fixmatch
            {"sweep": "e1pynkrn"},  # st++ full
            {"sweep": "zy0xrbns"},  # multiorgan, st, fixmatch, meanteacher
        ]
    },
    "test all": {
        "$or": [
            {"sweep": "xqffwt68"},  # test all
            {"sweep": "t7yc3a60"},  # Supervised all correct St++ weak augmentations
            {"sweep": "v24zcssb"},  # hippocampus, mean,st++,fixmatch
        ]
    },
}
# download data
api = wandb.Api()
runs = api.runs(f"gkeppler/{project_name}", filters=filters[name])
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
print("Amount of experiments BEFORE preprocessing: ", len(all_df))
# preprocessing
all_df["split"] = all_df["split"].apply(lambda x: x.replace("_", "/"))

# check if multiple dataset are in the sweep
if len(all_df["dataset"].unique()) > 1:
    print("more that on dataset in the metrics: ", all_df["dataset"].unique())

# drop rows with na in column "test mDICE"

all_df = all_df.dropna(subset=["test medpy_dc"])
print("Amount of experiments AFTER preprocessing: ", len(all_df))
metric_dict = {
    "mean DSC": "test medpy_dc",
    "mean IoU": "test medpy_jc",
    "mean Pixel Accuracy": "test overall_acc",
    "mean Average Surface Distance": "test medpy_asd",
    "mean Hausdorff Distance": "test medpy_hd",
    "runtime in sec": "_runtime",
}  # "test mDICE"#"test mIOU"
dataset_dict = {
    "melanoma": "ISIC Melanoma 2017",
    "breastCancer": "Breast Ultrasound",
    "pneumothorax": "Pneumothorax",
    "hippocampus": "Hippocampus",
    "zebrafish": "HeartSeg",
    "multiorgan": "Synapse multi-organ",
}
method_dict = {
    "Supervised": "Supervised",
    "CCT": "CCT",
    "St++": "ST++",
    "FixMatch": "FixMatch",
    "MeanTeacher": "MeanTeacher",
}
#%%
# get earlier values:
# the first entry in "test medpy_dc" is the test value computed with the checkpoint from the last epoch
metrics = ["test medpy_dc"]
list_metrics = []
for i, run in enumerate(runs):
    for j, row in run.history(keys=metrics).iterrows():
        # get only first elements of history -> supervised metric
        row.name = i
        list_metrics.append(row)
        break
late_epoch_df = pd.DataFrame(list_metrics).add_suffix(" lastepoch")
# merge with all_df
all_df = pd.concat([all_df, late_epoch_df], axis=1)

#%%
# drop duplicates
df = all_df.copy()
all_df = all_df.loc[all_df[["split", "dataset", "method", "shuffle"]].drop_duplicates(inplace=False).index, :]
print("Amount of experiments AFTER removing duplicates: ", len(all_df))

# %%
# dataset comparison
prep_df = all_df
metric_name = "runtime in sec"  # "mean DSC"
metric = metric_dict[metric_name]
colors = {"Supervised": "k", "St++": "b", "MeanTeacher": "g", "FixMatch": "y", "CCT": "r"}
check_df = pd.DataFrame()
dsc_df = pd.DataFrame()
f, axes = plt.subplots(2, 3, figsize=(12, 6))
# remove spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.4)
i = 0
for dataset_name, prep_df in all_df.groupby("dataset"):
    columns = [
        "split",
        "method",
        metric,
    ]
    method_df = prep_df[columns].groupby(["method", "split"]).agg([np.mean, np.std, np.count_nonzero])
    # if an entry of count_nonzero is not 5, print the dataset name:

    method_df.dropna(axis=0, how="all", inplace=True)
    columns_table = list(set([el[0] for el in method_df.columns.tolist()]))
    check_df = check_df.append(method_df[(metric, "count_nonzero")].rename(dataset_name))
    # draw figure
    # fig, axes = plt.subplots()
    ax = axes[i // 3, i % 3]
    i += 1
    for method, group in method_df.reset_index().groupby("method"):
        print(group.head())
        # add label to legend
        group.columns = ["".join(col) for col in group.columns]
        mean_list = group[metric + "mean"].values / 60  # * 100.0
        std_list = group[metric + "std"].values / 60  # * 100.0
        dsc_df.loc[method_dict[method], dataset_name + " " + group["split"]] = [
            f"{mean:.0f}" + "±" + f"{std:.0f}" for mean, std in zip(mean_list, std_list)
        ]
        # add a stepped horizontal lien to the plot
        if method == "Supervised":
            # get value of metric where split is 1
            value = group.loc[group["split"] == "1", metric + "mean"].values[0]
            print(value)
            # drop this row from the dataframe "group"

            group = group.drop(group[group["split"] == "1"].index)
            ax.axhline(y=value, color="k", linestyle="--", label="Full Supervised")

        # replace Na with 0
        group[metric + "std"].replace(np.NaN, 0, inplace=True)
        group["order"] = group["split"].str.split("/").apply(lambda x: float(x[0]) / float(x[1]) if len(x) > 1 else 1)
        print(group.head())
        group.sort_values(by="order").plot(
            "split", metric + "mean", yerr=metric + "std", label=method, ax=ax, color=colors[method]
        )
    ax.set_xlabel("Split")
    ax.set_ylabel(metric_name)
    ax.title.set_text(dataset_dict[dataset_name])
    # disable axis legend
    ax.get_legend().remove()
    # plt.legend()
    # legend without overlap to axes
plt.legend(loc="upper left", bbox_to_anchor=(1, 2.5), ncol=1, title="Methods")

if not os.path.exists("figures"):
    os.makedirs("figures")
plt.savefig(f"figures/{project_name}_{name}_{metric}_dataset_comparison.pdf", transparent=False, bbox_inches="tight")
print(check_df)
dsc_df = dsc_df.transpose().reset_index()
dsc_df.replace(np.NaN, "-", inplace=True)
dsc_df[["Dataset", "Split"]] = dsc_df["index"].str.split(" ", 1, expand=True)
# change name of entries in column "Dataset" using the dictionary
dsc_df["Dataset"] = dsc_df["Dataset"].map(dataset_dict)
dsc_df["order"] = dsc_df["Split"].str.split("/").apply(lambda x: float(x[0]) / float(x[1]) if len(x) > 1 else 1)
dsc_df = dsc_df.groupby("Dataset").apply(lambda x: x.sort_values(by="order", ascending=False))
dsc_df.drop("Dataset", axis=1, inplace=True)
dsc_df.drop("order", axis=1, inplace=True)
dsc_df.drop("index", axis=1, inplace=True)
dsc_df.reset_index(inplace=True)
dsc_df.set_index(["Dataset", "Split"], inplace=True)
dsc_df.drop("level_1", axis=1, inplace=True)
dsc_df
# %%
# calulated the realitve difference between the mean of the two methods and the Supervised method
difference = dsc_df.applymap(lambda x: x.replace("-", "NaN").split("±")[0]).astype(float)
# subtract series from every column in the dataframe
difference = difference.apply(lambda x: x - difference["Supervised"])
difference.replace(np.NaN, "-", inplace=True)
difference.drop("Supervised", axis=1, inplace=True)
difference
# %%
# to latex
print(dsc_df.transpose().to_latex(bold_rows=True))
# %%
# a graph about the DSC and trainer/global_step of the different methods
metric_name = "mean DSC"
metric = metric_dict[metric_name]
colors = {"Supervised": "k", "St++": "b", "MeanTeacher": "g", "FixMatch": "y", "CCT": "r"}
markers = {"1": "o", "1/4": "s", "1/8": "^", "1/30": "v"}
f, axes = plt.subplots(2, 3, figsize=(12, 6))
# remove spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.4)
i = 0
runtime_df = pd.DataFrame()
for dataset_name, prep_df in all_df.groupby("dataset"):
    columns = ["split", "method", metric, "_runtime"]
    method_df = prep_df[columns].groupby(["method", "split", "_runtime"]).agg([np.mean, np.std, np.count_nonzero])
    method_df.dropna(axis=0, how="all", inplace=True)
    columns_table = list(set([el[0] for el in method_df.columns.tolist()]))
    ax = axes[i // 3, i % 3]
    i += 1
    for label, group in method_df.reset_index().groupby(["method", "split"]):
        group.columns = ["".join(col) for col in group.columns]
        group.plot(
            "_runtime", metric + "mean", color=colors[label[0]], marker=markers[label[1]], ax=ax, kind="scatter"
        )
    ax.set_xlabel("Steps")
    ax.set_ylabel(metric_name)
    ax.title.set_text(dataset_dict[dataset_name])
    # ax.get_legend().remove()

legend1 = [mpatches.Patch(facecolor=list(colors.values())[i]) for i in range(5)]
legend2 = [plt.plot([], [], list(markers.values())[i], markerfacecolor="w", markeredgecolor="k")[0] for i in range(4)]
plt.legend(
    legend1 + legend2,
    list(colors.keys()) + list(markers.keys()),
    ncol=2,
    loc="upper left",
    bbox_to_anchor=(1, 2.5),
    title="Methods/Splits",
)

plt.savefig(f"figures/{project_name}_{name}_computational_requirements.pdf", transparent=False, bbox_inches="tight")

#%%
# calculate interpolated integral for each plot
# reduce all splits to one value
methods_list = ["MeanTeacher", "St++", "FixMatch", "CCT", "Supervised"]
dsc_interpol_df, runtime_mean_df = pd.DataFrame(), pd.DataFrame()
for dataset_name, prep_df in all_df.groupby("dataset"):
    for method in methods_list:
        metric = "_runtime"  # "test medpy_dc"  # "test medpy_dc lastepoch"
        columns = ["split", "_runtime", "test medpy_dc"]
        sup_df = prep_df[prep_df["method"] == "Supervised"]
        sup_df = sup_df[sup_df["split"] == "1"]
        method_df = prep_df[prep_df["method"] == method]
        # combine both dataframes
        comb_df = pd.concat([sup_df, method_df], axis=0)
        comb_df = comb_df[columns].groupby("split").agg([np.mean, np.std, np.count_nonzero])
        comb_df.columns = ["".join(col) for col in comb_df.columns]
        comb_df = comb_df.reset_index()[["split", "test medpy_dcmean", "_runtimemean"]]
        comb_df["split_value"] = (
            comb_df["split"].str.split("/").apply(lambda x: float(x[0]) / float(x[1]) if len(x) > 1 else 1)
        )
        comb_df = comb_df.sort_values("split_value")
        # print(dataset_name, method, comb_df)
        x, y = comb_df["split_value"].values, comb_df["test medpy_dcmean"].values

        # make spline derivative ~0.0 at end of x
        # x = np.append(x, 1.1)
        # y = np.append(y, y[-1])
        # spl = UnivariateSpline(x, y, k=3)
        # plt.plot(x, y, "ro", ms=5)
        # xs = np.linspace(0.03334, 1, 1000)
        # plt.plot(xs, spl(xs), "b", lw=3)
        # print("The qubic Integral is: ", spl.integral(0.0333, 1) / comb_df["test medpy_dcmean"].iloc[-1])

        # # linear interpolation
        try:
            f = interp1d(x, y)
        except Exception as e:
            print(e)
            continue
        # plt.plot(x, y, "ro", ms=5)
        # xs = np.linspace(0.03334, 1, 1000)
        # plt.plot(xs, f(xs), "g", lw=3)
        # # set x axit description
        # plt.xlabel("Split")
        # plt.ylabel(metric)
        # print("Linear interpoaltion: ",quad(f, 0.0334, 1)[0] / comb_df["test medpy_dcmean"].iloc[-1])
        dsc_interpol = 1.0  # quad(f, 0.0334, 1)[0] / comb_df["test medpy_dcmean"].iloc[-1]/(1-0.0334)
        runtime_mean = np.nanmean(comb_df["_runtimemean"].values)
        dsc_interpol_df.loc[dataset_dict[dataset_name], method_dict[method]] = f"{dsc_interpol:.2f}"
        runtime_mean_df.loc[dataset_dict[dataset_name], method_dict[method]] = f"{runtime_mean:.2f}"

runtime_mean_df = runtime_mean_df.apply(
    lambda x: runtime_mean_df["Supervised"].astype(float) / x.astype(float), axis=0
)
runtime_mean_df
#%%
# compute the mean per row
mean_list, std_list = dsc_interpol_df.astype(float).mean().values, dsc_interpol_df.astype(float).std().values
dsc_interpol_df = dsc_interpol_df.astype(float).applymap(lambda x: f"{x:.2f}")
dsc_interpol_df.loc[r"$Q_{Segmentation Quality}$"] = [
    f"{mean:.2f}" + "±" + f"{std:.2f}" for mean, std in zip(mean_list, std_list)
]

mean_list, std_list = runtime_mean_df.astype(float).mean().values, runtime_mean_df.astype(float).std().values
runtime_mean_df = runtime_mean_df.astype(float).applymap(lambda x: f"{x:.2f}")
runtime_mean_df.loc[r"$Q_{Time}$"] = [f"{mean:.2f}" + "±" + f"{std:.2f}" for mean, std in zip(mean_list, std_list)]

#%%
print(dsc_interpol_df.to_latex(bold_rows=True))
print(runtime_mean_df.to_latex(bold_rows=True))
# %%
# example of how to interpolate
x, y = np.array([0.0333, 0.125, 0.25, 1]), np.array([0.3, 0.55, 0.74, 0.8])
spl2 = UnivariateSpline(x, y, k=3)
spl2.set_smoothing_factor(0.5)
print(spl2.integral(0.0333, 1) / 0.96669)
# add markers to x-axis for each split
for i, xi in enumerate(x):
    plt.plot([xi, xi], [0, y[i]], "k--")
    # add label to x-axis
    # plt.text(xi, -0.04, comb_df["split"].iloc[i], ha="center", va="bottom")
xs = np.linspace(0.0333, 1, 1000)
plt.plot(xs, spl2(xs), "g", lw=3)
f = interp1d(x, y)
xs = np.linspace(0.03334, 1, 1000)
plt.plot(xs, f(xs), "b", lw=3)
plt.plot(x, y, "ro", ms=5)
# color the integral belwo the blue curve with light orange

plt.fill_between(xs, f(xs), 0.0, color="orange", alpha=0.2)
# Gamma as greek symbol in x lable
plt.xlabel(r"Splitratio $\Gamma$")
# text with unterscore Q_p
plt.ylabel(r"Metric e.g. DSC")
# horizontal line at 0.5
plt.plot([0.03, 1], [0.0, 0.0], "k--")
plt.savefig("metric_interpolation.pdf", pad_inches=0, bbox_inches="tight", transparent=True, dpi=300)
#
# %%
# NOT USED
# show metrics for method
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
prep_df = all_df
# ['brats':fehler, 'hippocampus':"1" unter "1/4", 'multiorgan':"super low",
#  'pneumothorax': low,'breastCancer': low,'zebrafish':"low", 'melanoma': ]
prep_df = prep_df[prep_df["dataset"] == "melanoma"]
for metric_name in metric_dict.keys():
    metric = metric_dict[metric_name]
    columns = [
        "split",
        "method",
        metric,
    ]
    method_df = prep_df[columns].groupby(["method", "split"]).agg([np.mean, np.std, np.count_nonzero])
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
            ax.axhline(y=value, color="k", linestyle="--", label="Full Supervised")

        # add label to legend

        group.columns = ["".join(col) for col in group.columns]
        # replace Na with 0
        group[metric + "std"].replace(np.NaN, 0, inplace=True)
        group["order"] = group["split"].str.split("/").apply(lambda x: float(x[0]) / float(x[1]) if len(x) > 1 else 1)
        print(group.head())
        group.sort_values(by="order").plot("split", metric + "mean", yerr=metric + "std", label=label, ax=ax)
    ax.set_xlabel("Split")
    ax.set_ylabel(metric_name)
    plt.legend()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    dataset = prep_df["dataset"].iloc[0]
    plt.savefig(
        f"figures/{project_name}_{name}_{dataset}_{metric}_methods.pdf", transparent=False, bbox_inches="tight"
    )
