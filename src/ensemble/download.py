"""
Download multi-model ensemble hindcasts from the Copernicus Climate Data
Store (CDS) and upload to Azure Blob Storage.

Usage
-----
    from src.ensemble.download import download_hindcasts

    download_hindcasts(
        issue_month=4,           # April
        valid_months=[7, 8, 9],  # JAS
        years=range(1993, 2017),
    )
"""

import io
import logging
import tempfile
from typing import Iterable

import cdsapi

from src.utils import blob_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centre → system-version mapping.  Bump the version number here when a new
# operational system is released; no other code needs to change.
# ---------------------------------------------------------------------------
CENTRE_SYSTEMS: dict[str, int] = {
    "ecmwf": 51,  # SEAS5.1
    "ukmo": 600,  # GloSea6
    "meteo_france": 8,  # System 8
    "dwd": 21,  # GCFS2.1
    "cmcc": 35,  # SPS3.5
    "ncep": 2,  # CFSv2
}

# Burkina Faso bounding box for the CDS area sub-selection [N, W, S, E]
BFA_AREA = [15.5, -6.0, 9.0, 3.0]

# Default hindcast period shared by most centres
DEFAULT_HINDCAST_YEARS = list(range(1993, 2017))


def _blob_name(centre: str, system: int, issue_month: int) -> str:
    return (
        f"{blob_utils.PROJECT_PREFIX}/raw/ensemble/"
        f"{centre}_i{issue_month:02d}_s{system}.nc"
    )


def _leadtimes_from_valid_months(
    issue_month: int, valid_months: Iterable[int]
) -> list[int]:
    """Convert absolute valid months to CDS lead-time integers."""
    lts = []
    for vm in valid_months:
        lt = vm - issue_month
        if lt <= 0:
            lt += 12
        lts.append(lt)
    return lts


def download_hindcasts(
    issue_month: int = 4,
    valid_months: Iterable[int] = (7, 8, 9),
    years: Iterable[int] = None,
    centres: Iterable[str] = None,
    stage: str = "dev",
    overwrite: bool = False,
) -> None:
    """
    Download hindcast ensemble means from CDS and upload to blob storage.

    One netCDF file per centre is produced, containing all requested years
    and the three valid months averaged into a single JAS total.

    Parameters
    ----------
    issue_month : int
        Forecast issue month (1=Jan … 12=Dec).  Default 4 (April).
    valid_months : iterable of int
        Months to treat as the valid season.  Default (7, 8, 9) = JAS.
    years : iterable of int, optional
        Hindcast years to download.  Defaults to DEFAULT_HINDCAST_YEARS.
    centres : iterable of str, optional
        Centres to download.  Defaults to all keys in CENTRE_SYSTEMS.
    stage : str
        Blob storage stage ('dev' or 'prod').
    overwrite : bool
        If False (default), skip centres whose blob already exists.
    """
    if years is None:
        years = DEFAULT_HINDCAST_YEARS
    if centres is None:
        centres = list(CENTRE_SYSTEMS.keys())

    years = sorted(years)
    valid_months = sorted(valid_months)
    leadtimes = _leadtimes_from_valid_months(issue_month, valid_months)

    client = cdsapi.Client()

    for centre in centres:
        system = CENTRE_SYSTEMS[centre]
        blob_name = _blob_name(centre, system, issue_month)

        if not overwrite:
            existing = blob_utils.list_container_blobs(
                name_starts_with=blob_name, stage=stage
            )
            if existing:
                logger.info("Skipping %s — blob already exists.", centre)
                continue

        logger.info(
            "Downloading %s (system %d) for issue month %02d …",
            centre,
            system,
            issue_month,
        )

        request = {
            "originating_centre": centre,
            "system": str(system),
            "variable": "total_precipitation",
            "product_type": "monthly_mean",
            "year": [str(y) for y in years],
            "month": [f"{issue_month:02d}"],
            "leadtime_month": [str(lt) for lt in leadtimes],
            "area": BFA_AREA,
            "format": "netcdf",
        }

        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            client.retrieve(
                "seasonal-original-single-levels", request, tmp.name
            )
            with open(tmp.name, "rb") as f:
                data = f.read()

        blob_utils._upload_blob_data(io.BytesIO(data), blob_name, stage=stage)
        logger.info("Uploaded %s to blob.", blob_name)
