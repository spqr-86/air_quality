from typing import List

import pandas as pd


def fix_dates(df: pd.core.frame.DataFrame, date_column: str) -> List[str]:
    """Fixes the date format in the dataframe.

    Args:
        df (pd.core.frame.DataFrame): The dataframe.
        date_column (str): Column with dates

    Returns:
        fixed_dates (List[str]): list of corrected dates to be
         put into the dataframe

    """
    dates = df[date_column]
    fixed_dates = []
    for row in dates:
        line = list(row)
        hour = int(''.join(line[11:13])) - 1
        fixed_dates.append(
            ''.join(line[:11] + [str(int(hour/10)) +
                                 str(int(hour % 10))] + line[13:]))
    return fixed_dates
