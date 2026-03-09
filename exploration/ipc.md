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

# IPC

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from src.datasources import ipc
from src.constants import *
```

```python
df_ipc = ipc.load_raw_ipc()
```

```python
df_ipc.dtypes
```

```python
df_ipc
```

```python
df_ipc.columns
```

```python
dff_proj = df_ipc[
    (df_ipc["adm1_pcod2"].isin(AOI_ADM1_PCODES))
    & (df_ipc["chtype"] == "projected")
    & (df_ipc["exercise_code"] == 1)
    & (df_ipc["reference_code"] == 3)
]
```

```python
dff_current = df_ipc[
    (df_ipc["adm1_pcod2"].isin(AOI_ADM1_PCODES))
    & (df_ipc["chtype"] == "current")
    & (df_ipc["exercise_code"] == 1)
]
```

```python
dff["exercise_label"].unique()
```

```python
dff["reference_code"].value_counts()
```

```python
dff["reference_label"].value_counts()
```

```python
dff["exercise_year"].unique()
```

```python
df_ipc_yearly_proj = (
    dff_proj.groupby("exercise_year")[["population", "phase35"]]
    .sum()
    .reset_index()
)
df_ipc_yearly_proj["frac35"] = (
    df_ipc_yearly_proj["phase35"] / df_ipc_yearly_proj["population"]
)
```

```python
df_ipc_yearly_current = (
    dff_current.groupby("exercise_year")[["population", "phase35"]]
    .sum()
    .reset_index()
)
df_ipc_yearly_current["frac35"] = (
    df_ipc_yearly_current["phase35"] / df_ipc_yearly_current["population"]
)
```

```python
df_ipc_yearly = df_ipc_yearly_current.merge(
    df_ipc_yearly_proj, on="exercise_year", suffixes=("_curr", "_proj")
)
```

```python
df_ipc_yearly
```

```python
df_ipc_yearly.plot(x="exercise_year", y="frac35_proj")
```

```python
df_ipc_yearly.plot(x="exercise_year", y="frac35_curr")
```
