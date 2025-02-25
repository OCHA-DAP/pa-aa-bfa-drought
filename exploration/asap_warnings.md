---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: pa-aa-bfa-drought
    language: python
    name: pa-aa-bfa-drought
---

# ASAP warnings

<!-- markdownlint-disable MD013 -->

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import calendar
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

from src.constants import *
from src.datasources import asap, seas5
from src.utils import dekad, blob_utils, rp_calc
```

```python
# asap.process_asap_warnings()
```

## Load pre-filtered data

And do some light processing

```python
df_asap = asap.load_processed_asap_warnings()
```

```python
def get_alert_gr_int(alert_gr_str):
    try:
        return int(alert_gr_str.removeprefix("Warning group "))
    except ValueError:
        return 0


for crop_range in ["crop", "range"]:
    df_asap[f"w_{crop_range}_gr_int"] = df_asap[f"w_{crop_range}_gr"].apply(
        get_alert_gr_int
    )

df_asap["year"] = df_asap["date"].dt.year
df_asap = df_asap[df_asap["year"] <= 2024].copy()
```

```python
df_asap[
    [x for x in df_asap.columns if "w_crop" in x]
].drop_duplicates().sort_values("w_crop")
```

Check that the "Warning with exceptional conditions" don't happen too often - looks fine.

```python
df_asap["w_crop"].value_counts().plot.bar()
```

```python
all_years = df_asap["date"].dt.year.unique()
```

```python
all_years
```

## Cycle through ASAP warning options

There's various ways to combine the warnings, but I broke it down to:

- either `OR` or `AND` condition to combine crop and range
- minimum number of admin1s with the alert level
- minimum alert level

```python
# options

crop_range_options = ["OR", "AND"]
minimum_adm1s_options = [1, 2, 3, 4]
alert_level_options = [1, 2, 3, 4]
```

```python
# set first possible trigger dekad to 3rd dekad of July
min_dekad = 21
# set last possible trigger dekad to 3rd dekad of July
max_dekad = 30

df_monitoring = df_asap[
    (df_asap["dekad"] >= min_dekad) & (df_asap["dekad"] <= max_dekad)
]

dfs = []
for crop_range in crop_range_options:
    for minimum_adm1s in minimum_adm1s_options:
        for alert_level in alert_level_options:
            dff = df_monitoring.copy()
            if crop_range == "OR":
                dff["indicator"] = dff[
                    ["w_crop_gr_int", "w_range_gr_int"]
                ].max(axis=1)
            elif crop_range == "AND":
                dff["indicator"] = dff[
                    ["w_crop_gr_int", "w_range_gr_int"]
                ].min(axis=1)
            else:
                raise ValueError("invalid crop_range")
            dff = dff[dff["indicator"] >= alert_level]
            adm_counts = (
                dff.groupby("year")
                .agg(
                    count_adm1s=("ADM1_PCODE", "nunique"),
                    min_dekad=("dekad", "min"),
                )
                .reset_index()
            )
            display(adm_counts)
            trigger_years = adm_counts[
                adm_counts["count_adm1s"] >= minimum_adm1s
            ][["year", "min_dekad"]]
            trigger_years[["crop_range", "minadm1s", "al"]] = (
                crop_range,
                minimum_adm1s,
                alert_level,
            )
            dfs.append(trigger_years)

df_triggers = pd.concat(dfs, ignore_index=True)
```

```python
df_triggers[
    (df_triggers["crop_range"] == "OR")
    & (df_triggers["minadm1s"] == 4)
    & (df_triggers["al"] == 4)
]
```

### Calculate return period

```python
df_rps = (
    df_triggers.groupby(["crop_range", "minadm1s", "al"])
    .agg(count=("year", "size"), min_dekad=("min_dekad", "mean"))
    .reset_index()
)
df_rps["rp"] = (len(all_years) + 1) / df_rps["count"]
df_rps = df_rps.sort_values("rp", ascending=False)
df_rps
```

### Plot RP of options

```python
lower_rp, upper_rp = 4, 20
```

```python
def plot_asap_heatmap(crop_range):
    bounds = [0, lower_rp, upper_rp, 1000]
    tick_bounds = bounds[1:-1]
    cmap = plt.cm.Spectral_r
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    crop_range_fr = "OU" if crop_range == "OR" else "ET"
    df_plot = df_rps[df_rps["crop_range"] == crop_range].pivot(
        values="rp", columns="minadm1s", index="al"
    )

    fig, ax = plt.subplots(dpi=200)

    sns.heatmap(df_plot, annot=True, cmap=cmap, norm=norm, ax=ax, cbar=False)

    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"Période de retour d'alertes ASAP\n(agricole {crop_range_fr} pâturage)"
    )
    ax.set_xlabel("Nombre de régions avec alerte")
    ax.set_ylabel("Niveau d'alerte minimum")
```

```python
plot_asap_heatmap("OR")
```

```python
plot_asap_heatmap("AND")
```

### Plot trigger timing of options

```python
def plot_asap_heatmap_min_dekad(crop_range):
    crop_range_fr = "OU" if crop_range == "OR" else "ET"
    df_plot = df_rps[df_rps["crop_range"] == crop_range].pivot(
        values="min_dekad", columns="minadm1s", index="al"
    )

    fig, ax = plt.subplots(dpi=200)

    sns.heatmap(
        df_plot, cmap="autumn_r", annot=True, ax=ax, cbar=False, fmt=".1f"
    )

    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"Première décade de déclenchement ASAP\n(agricole {crop_range_fr} pâturage)"
    )
    ax.set_xlabel("Nombre de régions avec alerte")
    ax.set_ylabel("Niveau d'alerte minimum")
```

```python
plot_asap_heatmap_min_dekad("OR")
```

```python
plot_asap_heatmap_min_dekad("AND")
```

```python
# set naming structure for ASAP triggers
ASAP_COL = (
    "Niveau ≥ {al}<br>N. régions ≥ {minadm1s}<br>" "Ag. {crop_range_fr} Pât."
)
```

```python
# set up reverse of naming structure
# in retropsect this could've all been avoided with proper grouping
# but this works for now
def extract_asap_params(formatted_string):
    # Define a regular expression pattern to match the expected format, allowing crop_range_fr to be a string
    pattern = r"Niveau ≥ (?P<al>\d+)<br>N\. régions ≥ (?P<minadm1s>\d+)<br>Ag\. (?P<crop_range_fr>[\w\s]+) Pât\."

    # Search the string for matches
    match = re.search(pattern, formatted_string)

    if match:
        # Extract the matched values
        al = match.group("al")
        minadm1s = match.group("minadm1s")
        crop_range_fr = match.group("crop_range_fr")
        return al, minadm1s, crop_range_fr
    else:
        raise ValueError("The string does not match the expected format")
```

### Check specific years triggered

```python
df_asap_yearly = pd.DataFrame(data={"year": range(2001, 2025)})
for _, row in df_rps.iterrows():
    crop_range_fr = "OU" if row["crop_range"] == "OR" else "ET"
    col_name = ASAP_COL.format(
        al=row["al"], minadm1s=row["minadm1s"], crop_range_fr=crop_range_fr
    )
    df_triggers_f = df_triggers[
        (df_triggers["crop_range"] == row["crop_range"])
        & (df_triggers["al"] == row["al"])
        & (df_triggers["minadm1s"] == row["minadm1s"])
    ]
    df_asap_yearly[col_name] = df_asap_yearly["year"].apply(
        lambda x: x in df_triggers_f["year"].unique()
    )
```

```python
crop_range
```

```python
df_triggers[
    (df_triggers["crop_range"] == crop_range)
    & (df_triggers["al"] == row["al"])
    & (df_triggers["minadm1s"] == row["minadm1s"])
]
```

```python
def highlight_true(value):
    if isinstance(value, bool) and value is True:
        return "background-color: crimson"
    else:
        return ""


def display_asap_activations(crop_range):
    dff = df_rps[
        (df_rps["rp"] >= lower_rp)
        & (df_rps["rp"] <= upper_rp)
        & (df_rps["crop_range"] == crop_range)
    ]

    df_disp = pd.DataFrame(data={"year": range(2001, 2025)})
    for _, row in dff.iterrows():
        col_name = f'Niveau ≥ {row["al"]}<br>N. régions ≥ {row["minadm1s"]}'
        df_triggers_f = df_triggers[
            (df_triggers["crop_range"] == crop_range)
            & (df_triggers["al"] == row["al"])
            & (df_triggers["minadm1s"] == row["minadm1s"])
        ]
        df_disp[col_name] = df_disp["year"].apply(
            lambda x: x in df_triggers_f["year"].unique()
        )

    display(
        df_disp.sort_values("year", ascending=False)
        .rename(columns={"year": "Année"})
        .set_index("Année")
        .style.map(highlight_true)
    )
```

```python
display_asap_activations("OR")
```

```python
display_asap_activations("AND")
```

## Combined RP

### Load SEAS5

```python
df_seas5 = seas5.load_seas5_stats(variable="rank")
```

```python
df_seas5
```

```python
df_seas5_yearly = df_seas5.pivot(
    index="year", columns="issued_month", values="q"
).reset_index()
df_seas5_yearly = df_seas5_yearly.rename(
    columns={x: f"issued_{x}" for x in [3, 7]}
)
```

### Merge with ASAP

```python
df_both_yearly = df_seas5_yearly.merge(df_asap_yearly).sort_values(
    "year", ascending=False
)
```

```python
df_both_yearly[ASAP_COL.format(al=4, crop_range_fr="OU", minadm1s=4)]
```

Quickly check the individual RP of SEAS5

```python
rp_based = False

rp_seas5 = 8
fixed_thresh = -0.75


for mo in [3, 7]:
    if rp_based:
        df_both_yearly[f"issued_{mo}_bool"] = df_both_yearly[
            f"issued_{mo}"
        ] < df_both_yearly[f"issued_{mo}"].quantile(1 / rp_seas5)
    else:
        df_both_yearly[f"issued_{mo}_bool"] = (
            df_both_yearly[f"issued_{mo}"] < fixed_thresh
        )
        print(f"issued month {mo}:")
        print((len(all_years) + 1) / df_both_yearly[f"issued_{mo}_bool"].sum())
        print()
```

```python
df_both_yearly[["year"] + [f"issued_{mo}_bool" for mo in [3, 7]]]
```

### Determine combined RPs

```python
asap_col = ASAP_COL.format(al=2, minadm1s=3, crop_range_fr="ET")

df_both_yearly.set_index("year")[
    ["issued_3_bool", "issued_7_bool", asap_col]
].style.map(highlight_true)
```

```python
for mo in [3, 7]:
    df_both_yearly = rp_calc.calculate_one_group_rp(
        df_both_yearly, col_name=f"issued_{mo}"
    )
```

```python
dicts = []

df_plot = df_both_yearly.copy()

asap_cols = [x for x in df_both_yearly.columns if "Niveau" in x]

for rank in range(len(df_plot)):
    df_plot["mar_trig"] = df_plot["issued_3_rank"] <= rank + 1
    df_plot["jul_trig"] = df_plot["issued_7_rank"] <= rank + 1
    rp_seas5_ind = (len(df_plot) + 1) / df_plot["mar_trig"].sum()
    df_plot["seas5_trig"] = df_plot[["mar_trig", "jul_trig"]].any(axis=1)
    rp_seas5_com = (len(df_plot) + 1) / df_plot["seas5_trig"].sum()
    for asap_col in asap_cols:
        rp_asap = (len(df_plot) + 1) / df_plot[asap_col].sum()
        df_plot[f"{asap_col}_any"] = df_plot[[asap_col, "seas5_trig"]].any(
            axis=1
        )
        rp_com = (len(df_plot) + 1) / df_plot[f"{asap_col}_any"].sum()
        if rp_com <= 5 and rp_com >= 3:
            dicts.append(
                {
                    "rp_com": rp_com,
                    "rp_asap": rp_asap,
                    "rp_seas5_com": rp_seas5_com,
                    "rp_seas5_ind": rp_seas5_ind,
                    "asap_col": asap_col,
                }
            )
```

```python
df_asap_v_seas5_rp = pd.DataFrame(dicts)
```

```python
df_asap_v_seas5_rp
```

```python
# this is literally just to make those little boxes on the plot
def plot_grid(x, y, symbol, ax, pitch=0.2):
    grid = np.zeros((2, 2))  # Create a 2x2 grid of empty boxes
    symbol = int(symbol)
    if symbol == 1:
        grid[0, 0] = 1  # Top-left
    elif symbol == 2:
        grid[0, 0] = 1  # Top-left
        grid[1, 1] = 1  # Bottom-right
    elif symbol == 3:
        grid[0, 0] = 1  # Top-left
        grid[1, 1] = 1  # Bottom-right
        grid[0, 1] = 1  # Top-right
    elif symbol == 4:
        grid[0, 0] = 1  # Top-left
        grid[0, 1] = 1  # Top-right
        grid[1, 0] = 1  # Bottom-left
        grid[1, 1] = 1  # Bottom-right

    ax.imshow(
        grid,
        extent=[x - pitch, x + pitch, y - pitch, y + pitch],
        origin="upper",
        cmap="Greys",
        alpha=1,
        vmin=0,
        vmax=1,
    )
    ax.plot(
        [x - pitch, x + pitch], [y - pitch, y - pitch], color="black", lw=0.2
    )  # Top border
    ax.plot(
        [x - pitch, x + pitch], [y + pitch, y + pitch], color="black", lw=0.2
    )  # Bottom border
    ax.plot(
        [x - pitch, x - pitch], [y - pitch, y + pitch], color="black", lw=0.2
    )  # Left border
    ax.plot(
        [x + pitch, x + pitch], [y - pitch, y + pitch], color="black", lw=0.2
    )
```

```python
# just to check which data points from the plot correspond to which ASAP triggers
df_asap_v_seas5_rp.sort_values(
    ["rp_com", "rp_asap", "rp_seas5_com", "rp_seas5_ind"], ascending=True
)
```

### Plot combined RP combinations

```python
fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

colors = ["dodgerblue", "green", "darkorange", "rebeccapurple"]

df_asap_v_seas5_rp_deduplicated = df_asap_v_seas5_rp.sort_values(
    "rp_com", ascending=True
).drop_duplicates(["rp_asap", "rp_seas5_ind"], keep="first")

for (rp_com, group), color in zip(
    df_asap_v_seas5_rp_deduplicated.groupby("rp_com"), colors
):
    group.plot(
        x="rp_seas5_ind",
        y="rp_asap",
        marker=".",
        linewidth=0,
        ax=ax,
        label=f"{rp_com:.1f}",
        color=color,
        markersize=20,
    )
    for _, row in group.iterrows():
        al, minadm1s, crop_range_fr = extract_asap_params(row["asap_col"])
        ax.annotate(
            "&" if crop_range_fr == "ET" else "||",
            (row["rp_seas5_ind"] + 0.35, row["rp_asap"]),
            ha="left",
            va="center",
            fontsize=6,
        )
        plot_grid(
            row["rp_seas5_ind"] + 0.85,
            row["rp_asap"],
            minadm1s,
            ax,
        )
        ax.annotate(
            al + "+",
            (row["rp_seas5_ind"] + 1.1, row["rp_asap"]),
            ha="left",
            va="center",
            fontsize=6,
        )

ax.legend(title="Période de retour\ncombinée (ans)")
ax.set_xlabel("Période de retour individuelle des prévisions (ans)")
ax.set_ylabel("Période de retour des alertes ASAP (ans)")
ax.set_title("Déclencheurs avec période de retour combinée acceptable")

lims = (3, 27)
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

### Display combined activations

```python
df_both_yearly
```

```python
def display_combined_activations(asap_col, rp_seas5_ind):
    df_disp = df_both_yearly.rename(columns={"year": "Année"}).set_index(
        "Année"
    )
    cols = []
    for mo in [3, 7]:
        col = f"Prévisions de<br>{FRENCH_MONTHS[calendar.month_abbr[mo]]}"
        df_disp[col] = df_disp[f"issued_{mo}_rp"] > rp_seas5_ind
        print(f"fcast {mo} rp:")
        print((len(df_disp) + 1) / df_disp[col].sum())
        print()
        cols.append(col)
    df_disp = df_disp[cols + [asap_col]]
    print(f"fcast combined rp:")
    print((len(df_disp) + 1) / df_disp[cols].any(axis=1).sum())
    print()
    print("asap rp:")
    print((len(df_disp) + 1) / df_disp[asap_col].sum())
    print()
    print("combined rp:")
    print((len(df_disp) + 1) / df_disp.any(axis=1).sum())
    display(df_disp.style.map(highlight_true))
```

```python
asap_col = ASAP_COL.format(al=4, minadm1s=1, crop_range_fr="ET")
display_combined_activations(asap_col, 8)
```

```python
asap_col = ASAP_COL.format(al=4, minadm1s=1, crop_range_fr="ET")
display_combined_activations(asap_col, 7)
```

```python
asap_col = ASAP_COL.format(al=4, minadm1s=3, crop_range_fr="ET")
display_combined_activations(asap_col, 5)
```

```python

```
