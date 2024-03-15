import os
from pathlib import Path

import xarray as xr

DATA_DIR = Path(os.getenv("AA_DATA_DIR_NEW"))
IRI_RAW_DIR = DATA_DIR / "public" / "raw" / "glb" / "iri"


def load_raw_iri():
    return xr.open_dataset(IRI_RAW_DIR / "iri.nc", decode_times=False)
