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

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from src.constants import *
from src.datasources import asap
from src.utils import dekad, blob_utils
```

```python
# asap.process_asap_warnings()
```

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

```python
df_asap["w_crop"].value_counts().plot.bar()
```

```python
all_years = df_asap["date"].dt.year.unique()
```

```python
all_years
```

```python
# options

crop_range_options = ["OR", "AND"]
minimum_adm1s_options = [1, 2, 3, 4]
alert_level_options = [1, 2, 3, 4]
biomass_only_options = [True, False]
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
            if biomass_only:
                dff = dff["indicator"]
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
df_triggers
```

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

```python
lower_rp, upper_rp = 3.5, 6
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

```python
def plot_asap_heatmap_min_dekad(crop_range):
    crop_range_fr = "OU" if crop_range == "OR" else "ET"
    df_plot = df_rps[df_rps["crop_range"] == crop_range].pivot(
        values="min_dekad", columns="minadm1s", index="al"
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        df_plot, cmap="autumn_r", annot=True, ax=ax, cbar=False, fmt=".1f"
    )

    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"Période de retour d'alertes ASAP\n(agricole {crop_range_fr} pâturage)"
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
df_rps
```

```python
df_rps_acceptable = df_rps[
    (df_rps["rp"] >= lower_rp) & (df_rps["rp"] <= upper_rp)
]
```

```python
df_rps_acceptable
```

```python
df_triggers
```

```python
def display_asap_activations(crop_range):
    dff = df_rps[
        (df_rps["rp"] >= lower_rp)
        & (df_rps["rp"] <= upper_rp)
        & (df_rps["crop_range"] == crop_range)
    ]

    def highlight_true(value):
        if isinstance(value, bool) and value is True:
            return "background-color: crimson"
        else:
            return ""

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

```python
df_yearly = pd.DataFrame(data={"year": range(2001, 2025)})
```

```python
df_yearly["f"] = df_yearly["year"].apply(lambda x: x > 2010)
```

```python
df_yearly
```

```python
df_yearly.index.values
```
