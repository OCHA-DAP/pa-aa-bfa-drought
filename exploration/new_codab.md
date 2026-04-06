---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: pa-aa-bfa-drought
    language: python
    name: pa-aa-bfa-drought
---

# New CODAB

As of 2026

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus

from src.utils.blob_utils import PROJECT_PREFIX
```

```python
adm1 = stratus.codab.load_codab_from_fieldmaps(iso3="bfa", admin_level=1)
```

```python
adm1
```

```python
# looks like the FieldMaps version is out of date
```

```python
blob_name = f"{PROJECT_PREFIX}/raw/codab/bfa_admin_boundaries.shp.zip"

adm1_new = stratus.load_shp_from_blob(blob_name, shapefile="bfa_admin1.shp")
```

```python
adm1_new.plot()
```

```python
adm1_new
```

```python
# this one is correct
```

```python
adm1_new.total_bounds
```
