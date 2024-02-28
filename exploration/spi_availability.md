---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: pa-aa-bfa-drought
    language: python
    name: pa-aa-bfa-drought
---

# SPI data availability

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import datetime
import requests

import pandas as pd
from tqdm.notebook import tqdm
```

```python
CPC_BASE_URL = "https://storage.googleapis.com/noaa-nidis-drought-gov-data/datasets/cog/v1"
```

```python
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date.today()
dates = pd.date_range(start_date, end_date, freq="D")
```

```python
dicts = []
for date in tqdm(dates):
    date_str = date.strftime("%Y-%m-%d")
    date_url = f"{CPC_BASE_URL}/{date.year}/{date_str}"
    era5_filename = f"GLOBAL-ERA5-spi-1mo-{date_str}.tif"
    cmorph_filename = f"GLOBAL-NOAA_CPC_DAILY_GLOBAL-spi-1mo-{date_str}.tif"
    era5_url = f"{date_url}/{era5_filename}"
    cmorph_url = f"{date_url}/{cmorph_filename}"
    era5_response = requests.head(era5_url)
    cmorph_response = requests.head(cmorph_url)
    dicts.append(
        {
            "date": date,
            "era5": era5_response.status_code == 200,
            "cmorph": cmorph_response.status_code == 200,
        }
    )

df = pd.DataFrame(dicts)
```

```python
df.groupby([df["date"].dt.year, df["date"].dt.month]).mean(numeric_only=True)
```

```python
df.iloc[-20:]
```
