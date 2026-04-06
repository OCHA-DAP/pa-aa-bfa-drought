from typing import Literal

import pandas as pd
from tqdm.auto import tqdm

from src.constants import ASAP0_ID, ASAP1_IDS, ASAP2_IDS
from src.utils import blob_utils, dekad

ASAP_BLOB_NAME = (
    blob_utils.PROJECT_PREFIX + "/{data_type}/asap/{specific_path}"
)


def get_blob_name(
    data_type: Literal["raw", "processed"],
    variable: Literal["warnings", "warnings_l2"],
):
    specific_path = None
    if data_type == "raw":
        if variable == "warnings":
            specific_path = "warnings_ts/warnings_ts.csv"
        elif variable == "warnings_l2":
            specific_path = "warnings_l2_ts/warnings_l2_ts.csv"
    elif data_type == "processed":
        if variable == "warnings":
            specific_path = "bfa_aoi_adm1_warnings.parquet"
        elif variable == "warnings_l2":
            specific_path = "bfa_aoi_adm2_warnings.parquet"
    if specific_path is None:
        raise ValueError(
            f"Invalid data type {data_type} and variable {variable}"
        )
    return ASAP_BLOB_NAME.format(
        data_type=data_type, specific_path=specific_path
    )


def process_asap_warnings():
    df = load_raw_asap_warnings()
    df_bfa = df[df["asap0_id"] == ASAP0_ID].copy()
    df_bfa["date"] = pd.to_datetime(df_bfa["date"])
    df_aoi = df_bfa[df_bfa["asap1_id"].isin(ASAP1_IDS.values())].copy()
    df_aoi["dekad"] = df_aoi["date"].apply(dekad.date_to_dekad)
    df_aoi["ADM1_PCODE"] = df_aoi["asap1_id"].replace(
        {v: k for k, v in ASAP1_IDS.items()}
    )
    blob_name = get_blob_name("processed", "warnings")
    blob_utils.upload_parquet_to_blob(df_aoi, blob_name)


def load_raw_asap_warnings():
    """Load raw ASAP warnings data from blob storage,
    which were downloaded from:
    https://agricultural-production-hotspots.ec.europa.eu/download.php
    (filename warnings_ts.zip/warnings_ts.csv)
    """
    blob_name = get_blob_name("raw", "warnings")
    return blob_utils.load_csv_from_blob(blob_name, sep=";")


def load_processed_asap_warnings():
    blob_name = get_blob_name("processed", "warnings")
    return blob_utils.load_parquet_from_blob(blob_name)


def process_asap_warnings_adm2(filepath: str):
    chunksize = 500_000
    with open(filepath, "rb") as f:
        n_rows = sum(1 for _ in f) - 1  # subtract header
    total_chunks = (n_rows + chunksize - 1) // chunksize
    asap2_ids = set(ASAP2_IDS.values())
    chunks = []
    for chunk in tqdm(
        pd.read_csv(filepath, sep=";", chunksize=chunksize), total=total_chunks
    ):
        filtered = chunk[
            (chunk["asap0_id"] == ASAP0_ID) & chunk["asap2_id"].isin(asap2_ids)
        ]
        if not filtered.empty:
            chunks.append(filtered)
    df_aoi = pd.concat(chunks).copy()
    df_aoi["date"] = pd.to_datetime(df_aoi["date"])
    df_aoi["dekad"] = df_aoi["date"].apply(dekad.date_to_dekad)
    df_aoi["ADM2_PCODE"] = df_aoi["asap2_id"].replace(
        {v: k for k, v in ASAP2_IDS.items()}
    )
    blob_name = get_blob_name("processed", "warnings_l2")
    blob_utils.upload_parquet_to_blob(df_aoi, blob_name)


def load_processed_asap_warnings_adm2():
    blob_name = get_blob_name("processed", "warnings_l2")
    return blob_utils.load_parquet_from_blob(blob_name)
