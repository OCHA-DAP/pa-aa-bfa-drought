from typing import Literal

from src.utils import blob_utils

ASAP_BLOB_NAME = (
    blob_utils.PROJECT_PREFIX + "/{data_type}/asap/{specific_path}"
)


def get_blob_name(
    data_type: Literal["raw", "processed"], variable: Literal["warnings"]
):
    specific_path = None
    if data_type == "raw":
        if data_type == "raw":
            if variable == "warnings":
                specific_path = "warnings_ts/warnings_ts.csv"
    if specific_path is None:
        raise ValueError(
            f"Invalid data type {data_type} and variable {variable}"
        )
    return ASAP_BLOB_NAME.format(
        data_type=data_type, specific_path=specific_path
    )


def load_raw_asap_warnings():
    blob_name = get_blob_name("raw", "warnings")
    return blob_utils.load_csv_from_blob(blob_name)
