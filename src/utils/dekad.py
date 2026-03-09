import pandas as pd


def date_to_dekad(date: pd.Timestamp) -> int:
    return (date.month - 1) * 3 + (date.day - 1) // 10 + 1
