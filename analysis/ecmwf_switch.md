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

# ECMWF switch

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from src.datasources import codab, seas5
```

```python
codab.download_codab_to_blob()
```

```python
adm1 = codab.load_codab_from_blob(admin_level=1, aoi_only=True)
```

```python
adm1.plot()
```

```python
da_seas5 = seas5.open_seas5_rasters()
```

```python
da_seas5
```
