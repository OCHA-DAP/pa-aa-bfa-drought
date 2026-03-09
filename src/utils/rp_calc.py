from typing import List

import pandas as pd


def calculate_one_group_rp(group, col_name: str = "q", ascending: bool = True):
    """Calculate the empirical RP for a single group.

    Parameters
    ----------
    group : pd.DataFrame
        The group for which to calculate the RP.
    col_name : str, optional
        The name of the column for which to calculate the RP, by default "q".
    ascending : bool, optional
        Whether to rank the column in ascending order, by default True.
        Should be False for cases where a high number is severe
        (e.g. precipitation for flooding), and True for cases where a low
        number is severe (e.g. precipitation for drought).

    Returns
    -------
    pd.DataFrame
        The input group with the RP columns added.
    """
    group[f"{col_name}_rank"] = group[col_name].rank(ascending=ascending)
    group[f"{col_name}_rp"] = (len(group) + 1) / group[f"{col_name}_rank"]
    return group


def calculate_groups_rp(df: pd.DataFrame, by: List):
    """Calculate the empirical RP for each group in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame for which to calculate the RP.
    by : List
        The columns by which to group the DataFrame.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with the RP columns added.
    """
    return (
        df.groupby(by)
        .apply(calculate_one_group_rp, include_groups=False)
        .reset_index()
        .drop(columns=f"level_{len(by)}")
    )
