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

from src.constants import *
from src.datasources import asap
from src.utils import dekad, blob_utils
```

```python
asap.process_asap_warnings()
```

```python
df_asap = asap.load_processed_asap_warnings()
```

```python
df_asap
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
jul_end_dekad = 21
aug_end_dekad = 24
sep_end_dekad = 27
```

```python
df_jul_end = df_asap[df_asap["dekad"] == jul_end_dekad]
```

```python
df_jul_end
```
