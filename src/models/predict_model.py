from typing import Dict

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def calculate_mae_for_nearest_station(df: pd.core.frame.DataFrame,
                                      target: str) -> Dict[str, float]:
    """Create a nearest neighbor model and run it on your test data.

    Args:
        df (pd.core.frame.DataFrame): The dataframe
        target (str): The chosen pollutant for which it plots the distribution

    """
    df2 = df.dropna(inplace=False)
    df2.insert(0, 'time_discriminator', (
        df2['DateTime'].dt.dayofyear * 100000 + df2[
            'DateTime'].dt.hour * 100).values, True)

    train_df, test_df = train_test_split(df2, test_size=0.2, random_state=57)

    imputer = KNNImputer(n_neighbors=1)
    imputer.fit(
        train_df[['time_discriminator', 'Latitude', 'Longitude', target]])

    # regression_scores = {}

    y_test = test_df[target].values

    test_df2 = test_df.copy()
    test_df2.loc[test_df.index, target] = float('NAN')

    y_pred = imputer.transform(
        test_df2[['time_discriminator', 'Latitude', 'Longitude', target]]
    )[:, 3]

    return {'MAE': mean_absolute_error(y_pred, y_test)}
