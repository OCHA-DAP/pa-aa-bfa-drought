from src.utils import blob_utils


def load_raw_ipc():
    blob_name = f"{blob_utils.PROJECT_PREFIX}/raw/ipc/cadre_harmonise_caf_ipc_mar24_final_ver-2.xlsx"  # noqa
    return blob_utils.load_excel_from_blob(blob_name)
