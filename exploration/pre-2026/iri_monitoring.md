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

# IRI monitoring

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from src.datasources import iri, codab
```

```python
adm2 = codab.load_codab()
ADM1_AOI_PCODES = ["BF46", "BF54", "BF56", "BF49"]
adm2_aoi = adm2[adm2["ADM1_PCODE"].isin(ADM1_AOI_PCODES)]
```

```python
adm2_aoi.plot()
```

```python
ds = iri.load_raw_iri()
```

```python

```
