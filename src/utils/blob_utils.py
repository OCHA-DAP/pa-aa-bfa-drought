import io
import os
import shutil
import tempfile
import zipfile
from typing import Literal

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import xarray as xr
from azure.storage.blob import ContainerClient, ContentSettings
from dotenv import load_dotenv

load_dotenv()

PROD_BLOB_SAS = os.getenv("DS_AZ_BLOB_PROD_SAS")
DEV_BLOB_SAS = os.getenv("DS_AZ_BLOB_DEV_SAS_WRITE")

DS_AZ_BLOB_DEV_HOST = "imb0chd0dev.blob.core.windows.net"
DS_AZ_BLOB_PROD_HOST = "imb0chd0prod.blob.core.windows.net"

AZURE_BLOB_BASE_URL = "https://{host}/{container_name}?{sas}"

PROJECT_PREFIX = "ds-aa-tcd-drought"


def get_container_client(
    container_name: str = "projects", stage: Literal["prod", "dev"] = "dev"
):
    """
    Get an Azure Blob Storage container client.

    Parameters
    ----------
    container_name : str, optional
        Name of the container to connect to, by default "projects"
    stage : Literal["prod", "dev"], optional
        Environment stage to connect to, by default "dev"

    Returns
    -------
    ContainerClient
        Azure storage container client object
    """
    if stage == "dev":
        url = AZURE_BLOB_BASE_URL.format(
            host=DS_AZ_BLOB_DEV_HOST,
            container_name=container_name,
            sas=DEV_BLOB_SAS,
        )
    elif stage == "prod":
        url = AZURE_BLOB_BASE_URL.format(
            host=DS_AZ_BLOB_PROD_HOST,
            container_name=container_name,
            sas=PROD_BLOB_SAS,
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")
    return ContainerClient.from_container_url(url)


def upload_parquet_to_blob(
    df,
    blob_name,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
    **kwargs,
):
    """
    Upload a pandas DataFrame to Azure Blob Storage in parquet format.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to upload
    blob_name : str
        Name of the blob to create/update
    stage : Literal["prod", "dev"], optional
        Environment stage to upload to, by default "dev"
    container_name : str, optional
        Name of the container to upload to, by default "projects"
    **kwargs : dict
        Additional arguments passed to pandas.DataFrame.to_parquet()
    """
    _upload_blob_data(
        df.to_parquet(**kwargs),
        blob_name,
        stage=stage,
        container_name=container_name,
    )


def load_parquet_from_blob(
    blob_name,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
):
    """
    Load a parquet file from Azure Blob Storage into a pandas DataFrame.

    Parameters
    ----------
    blob_name : str
        Name of the blob to load
    stage : Literal["prod", "dev"], optional
        Environment stage to load from, by default "dev"
    container_name : str, optional
        Name of the container to load from, by default "projects"

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the loaded data
    """
    blob_data = _load_blob_data(
        blob_name, stage=stage, container_name=container_name
    )
    return pd.read_parquet(io.BytesIO(blob_data))


def upload_csv_to_blob(
    df,
    blob_name,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
    **kwargs,
):
    """
    Upload a pandas DataFrame to Azure Blob Storage in CSV format.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to upload
    blob_name : str
        Name of the blob to create/update
    stage : Literal["prod", "dev"], optional
        Environment stage to upload to, by default "dev"
    container_name : str, optional
        Name of the container to upload to, by default "projects"
    **kwargs : dict
        Additional arguments passed to pandas.DataFrame.to_csv()
    """
    _upload_blob_data(
        df.to_csv(index=False, **kwargs),
        blob_name,
        stage=stage,
        content_type="text/csv",
        container_name=container_name,
    )


def load_csv_from_blob(
    blob_name,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
    **kwargs,
):
    """
    Load a CSV file from Azure Blob Storage into a pandas DataFrame.

    Parameters
    ----------
    blob_name : str
        Name of the blob to load
    stage : Literal["prod", "dev"], optional
        Environment stage to load from, by default "dev"
    container_name : str, optional
        Name of the container to load from, by default "projects"
    **kwargs : dict
        Additional arguments passed to pandas.read_csv()

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the loaded data
    """
    blob_data = _load_blob_data(
        blob_name, stage=stage, container_name=container_name
    )
    return pd.read_csv(io.BytesIO(blob_data), **kwargs)


def upload_shp_to_blob(gdf, blob_name, stage: Literal["prod", "dev"] = "dev"):
    """
    Upload a GeoDataFrame to Azure Blob Storage as a zipped shapefile.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to upload
    blob_name : str
        Name of the blob to create/update
    stage : Literal["prod", "dev"], optional
        Environment stage to upload to, by default "dev"
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # File paths for shapefile components within the temp directory
        shp_base_path = os.path.join(temp_dir, "data")

        gdf.to_file(shp_base_path, driver="ESRI Shapefile")

        zip_file_path = os.path.join(temp_dir, "data")

        shutil.make_archive(
            base_name=zip_file_path, format="zip", root_dir=temp_dir
        )

        # Define the full path to the zip file
        full_zip_path = f"{zip_file_path}.zip"

        # Upload the buffer content as a blob
        with open(full_zip_path, "rb") as data:
            _upload_blob_data(data, blob_name, stage=stage)


# TODO: Allow for specification of local directory
def load_shp_from_blob(
    blob_name, shapefile: str = None, stage: Literal["prod", "dev"] = "dev"
):
    """
    Load a zipped shapefile from Azure Blob Storage into a GeoDataFrame.

    Parameters
    ----------
    blob_name : str
        Name of the blob to load
    shapefile : str, optional
        Name of the specific shapefile within the zip to load, by default None
    stage : Literal["prod", "dev"], optional
        Environment stage to load from, by default "dev"

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the loaded spatial data
    """
    blob_data = _load_blob_data(blob_name, stage=stage)
    with zipfile.ZipFile(io.BytesIO(blob_data), "r") as zip_ref:
        zip_ref.extractall("temp")
        if shapefile is None:
            shapefile = [f for f in zip_ref.namelist() if f.endswith(".shp")][
                0
            ]
        gdf = gpd.read_file(f"temp/{shapefile}")
    return gdf


def _load_blob_data(
    blob_name,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
):
    """
    Internal function to load raw data from a blob.

    Parameters
    ----------
    blob_name : str
        Name of the blob to load
    stage : Literal["prod", "dev"], optional
        Environment stage to load from, by default "dev"
    container_name : str, optional
        Name of the container to load from, by default "projects"

    Returns
    -------
    bytes
        Raw blob data
    """
    container_client = get_container_client(
        stage=stage, container_name=container_name
    )
    blob_client = container_client.get_blob_client(blob_name)
    data = blob_client.download_blob().readall()
    return data


def _upload_blob_data(
    data,
    blob_name,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
    content_type: str = None,
):
    """
    Internal function to upload raw data to Azure Blob Storage.

    Parameters
    ----------
    data : bytes or BinaryIO
        Data to upload
    blob_name : str
        Name of the blob to create/update
    stage : Literal["prod", "dev"], optional
        Environment stage to upload to, by default "dev"
    container_name : str, optional
        Name of the container to upload to, by default "projects"
    content_type : str, optional
        MIME type of the content, by default None
    """
    container_client = get_container_client(
        stage=stage, container_name=container_name
    )

    if content_type is None:
        content_settings = ContentSettings(
            content_type="application/octet-stream"
        )
    else:
        content_settings = ContentSettings(content_type=content_type)

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        data, overwrite=True, content_settings=content_settings
    )


def list_container_blobs(
    name_starts_with=None,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
):
    """
    List all blobs in a container with optional prefix filtering.

    Parameters
    ----------
    name_starts_with : str, optional
        Prefix to filter blob names, by default None
    stage : Literal["prod", "dev"], optional
        Environment stage to list from, by default "dev"
    container_name : str, optional
        Name of the container to list from, by default "projects"

    Returns
    -------
    list
        List of blob names in the container
    """
    container_client = get_container_client(
        stage=stage, container_name=container_name
    )
    return [
        blob.name
        for blob in container_client.list_blobs(
            name_starts_with=name_starts_with
        )
    ]


def _get_blob_url(
    blob_name,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
):
    """
    Get the URL for a blob in Azure Storage.

    Parameters
    ----------
    blob_name : str
        Name of the blob
    stage : Literal["prod", "dev"], optional
        Environment stage, by default "dev"
    container_name : str, optional
        Name of the container, by default "projects"

    Returns
    -------
    str
        Complete URL to access the blob
    """
    container_client = get_container_client(
        stage=stage, container_name=container_name
    )
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.url


def open_blob_cog(
    blob_name,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
    chunks=None,
):
    """
    Open a Cloud Optimized GeoTIFF (COG) from Azure Blob Storage.

    Parameters
    ----------
    blob_name : str
        Name of the COG blob
    stage : Literal["prod", "dev"], optional
        Environment stage, by default "dev"
    container_name : str, optional
        Name of the container, by default "projects"
    chunks : bool or dict, optional
        Chunk size for dask array, by default None

    Returns
    -------
    xarray.DataArray
        DataArray containing the raster data
    """
    cog_url = _get_blob_url(
        blob_name, stage=stage, container_name=container_name
    )
    if chunks is None:
        chunks = True
    return rxr.open_rasterio(cog_url, chunks=chunks)


def upload_cog_to_blob(
    da: xr.DataArray,
    blob_name: str,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
):
    """
    Upload an xarray DataArray as a Cloud Optimized GeoTIFF (COG)
    to Azure Blob Storage.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing the raster data to upload
    blob_name : str
        Name of the blob to create/update
    stage : Literal["prod", "dev"], optional
        Environment stage to upload to, by default "dev"
    container_name : str, optional
        Name of the container to upload to, by default "projects"
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmpfile:
        temp_filename = tmpfile.name
        da.rio.to_raster(temp_filename, driver="COG")
        with open(temp_filename, "rb") as f:
            get_container_client(
                container_name=container_name, stage=stage
            ).get_blob_client(blob_name).upload_blob(f, overwrite=True)
