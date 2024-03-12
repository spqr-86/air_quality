from typing import Dict

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def calculate_mae_for_nearest_station(df: pd.core.frame.DataFrame,
                                      target: str) -> Dict[str, float]:
    """Create a nearest neighbor model and run it on your test data.

    Args:
        df (pd.core.frame.DataFrame): The dataframe
        target (str): The chosen pollutant for which it plots the distribution

    """
    df2 = df.dropna(inplace=False)
    df2.insert(0,
               'time_discriminator',
               (df2['DateTime'].dt.dayofyear * 100000 +
                df2['DateTime'].dt.hour * 100).values,
               True
               )

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


def calculate_mae_for_k(
        data: pd.core.frame.DataFrame,
        k: int = 1,
        target_pollutant: str = 'PM2.5'
) -> float:
    """Calculates the MAE for k nearest neighbors.

    Args:
        data (pd.core.frame.DataFrame): dataframe with data.
        k (int): number of neighbors to use for interpolation
        target_pollutant (str): pollutant for which to show the heatmap

    Returns:
        MAE (float): The MAE value

    """
    # Drop all the rows with the stations where the data imputation didnt
    # perform well
    bad_stations = ['7MA', 'CSE', 'COL', 'MOV2']
    df2 = data.drop(data[data['Station'].isin(bad_stations)].index)

    # Drop all the rows where there is imputed data, so the calculation
    # is only done on real data
    # df2 = data[data[[c for c in data.columns if
    # 'flag' in c]].isnull().all(axis=1)]

    # Take a sample of the data (so that the notebook runs faster)
    df2 = df2.sample(frac=0.2, random_state=8765)
    df2.insert(0,
               'time_discriminator',
               (df2['DateTime'].dt.dayofyear * 10000 +
                df2['DateTime'].dt.hour * 100).values,
               True
               )
    predictions = []
    stations = data['Station'].unique()
    for station in stations:
        df_day_station = df2.loc[df2['Station'] == station]
        if len(df_day_station) > 0:
            df_day_no_station = df2.loc[df2['Station'] != station]
            if len(df_day_no_station) >= k:
                neigh = KNeighborsRegressor(n_neighbors=k, weights='distance',
                                            metric='sqeuclidean')
                knn_model = neigh.fit(
                    df_day_no_station[
                        ['Latitude', 'Longitude', 'time_discriminator']],
                    df_day_no_station[[target_pollutant]]
                )
                prediction = knn_model.predict(df_day_station[
                    ['Latitude', 'Longitude',
                     'time_discriminator']])
                if len(predictions) == 0:
                    predictions = np.array(
                        [df_day_station[target_pollutant].values,
                         prediction[:, 0]]).T
                else:
                    predictions = np.concatenate(
                        (predictions, np.array(
                            [df_day_station[target_pollutant].values,
                             prediction[:, 0]]).T),
                        axis=0
                    )

    predictions = np.array(predictions)
    mae = mean_absolute_error(predictions[:, 0], predictions[:, 1])

    return mae
