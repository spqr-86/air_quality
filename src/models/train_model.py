import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# This is a list of all pollutants
pollutants_list = ['PM2.5', 'PM10',  'NO', 'NO2', 'NOX', 'CO', 'OZONE']


def build_keras_model(input_size: int) -> tf.keras.Model:
    """Build a neural network with three fully connected
    layers (sizes: 64, 32, 1)

    Args:
        input_size (int): The size of the input

    Returns:
        model (tf.keras.Model): The neural network
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_size]),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.007)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])

    return model


def train_and_test_model(
        feature_names: List[str],
        target: str,
        train_df: pd.core.frame.DataFrame,
        test_df: pd.core.frame.DataFrame,
        model: tf.keras.Model,
        number_epochs: int = 100
) -> Tuple[tf.keras.Model, StandardScaler, Dict[str, float]]:
    """This function will take the features (x), the target (y) and the model
    and will fit and Evaluate the model.

    Args:
        feature_names (List[str]): Names of feature columns
        target (str): Name of the target column
        train_df (pd.core.frame.DataFrame): Dataframe with training data
        test_df (pd.core.frame.DataFrame): Dataframe with test data
        model (tf.keras.Model): Model to be fit to the data
        number_epochs (int): Number of epochs

    Returns:
        model (tf.keras.Model): Fitted model
        scaler (StandardScaler): scaler
        MAE (Dict[str, float]): Dictionary containing mean absolute error.

    """
    scaler = StandardScaler()

    x_train = train_df[feature_names]
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    y_train = train_df[target]
    x_test = test_df[feature_names]
    x_test = scaler.transform(x_test)
    y_test = test_df[target]

    # Build and train model
    model.fit(x_train, y_train, batch_size=64, epochs=number_epochs)
    y_pred = model.predict(x_test)
    # print(f"\nModel Score: {model.score(X_test, y_test)}")
    mae = {'MAE': mean_absolute_error(y_pred, y_test)}
    return model, scaler, mae


def impute_nontarget_missing_values_interpolate(
        df_with_missing: pd.core.frame.DataFrame,
        feature_names: List[str],
        target: str,
) -> pd.core.frame.DataFrame:
    """
    Imputes data to non-target variables using interpolation.
    This data can then be used by NN to impute the target column.

    Args:
        df_with_missing (pd.core.frame.DataFrame): The dataframe with the data.
        feature_names (List[str]): Names of feature columns
        target (str): Name of the target column

    Returns:
        imputed_values_with_flag (pd.core.frame.DataFrame): The dataframe with
        imputed values and flags.
    """
    pollutants_except_target = [i for i in pollutants_list if i != target]

    # Flag the data that was imputed
    imputed_flag = df_with_missing[pollutants_except_target]

    warnings.filterwarnings('ignore')

    for pollutant in pollutants_except_target:
        # Create the flag column for the pollutant
        imputed_flag[f'{pollutant}_imputed_flag'] = np.where(
            imputed_flag[pollutant].isnull(), 'interpolated', None)
        imputed_flag.drop(pollutant, axis=1, inplace=True)

        # Impute a value to the first one if it is missing, because
        # interpolate does not fix the first value
        if np.any(df_with_missing.loc[
                [df_with_missing.index[0]], [pollutant]].isnull()):
            df_with_missing.loc[[df_with_missing.index[0]], [pollutant]] = [12]

    # Interpolate missing values
    imputed_values = df_with_missing[feature_names].interpolate(
        method='linear')

    imputed_values_with_flag = imputed_values.join(imputed_flag)

    return imputed_values_with_flag


def impute_target_missing_values_neural_network(
        df_with_missing: pd.core.frame.DataFrame,
        model: tf.keras.Model,
        scaler: StandardScaler,
        baseline_imputed: pd.core.frame.DataFrame,
        target: str,
) -> pd.core.frame.DataFrame:
    """
    Imputes data to non-target variables using interpolation.
    This data can then be used by NN to impute the target column.

    Args:
        df_with_missing (pd.core.frame.DataFrame): The dataframe with the data.
        model (tf.keras.Model): Model
        scaler (StandardScaler): scaler
        baseline_imputed (pd.core.frame.DataFrame): The dataframe with imputed
        values and flags for nontarget.
        target (str): Name of the target column

    Returns:
        data_with_imputed (pd.core.frame.DataFrame): The dataframe with
        imputed values and flags.
    """
    # Metadata columns that we want to output in the end
    metadata_columns = ['DateTime', 'Station', 'Latitude', 'Longitude']

    # Save the data and imputed flags of nontarget pollutant for
    # outputting later
    baseline_imputed_data_and_flags = baseline_imputed[
        [i for i in list(baseline_imputed.columns) if
         i in pollutants_list or 'flag' in i]]

    # Flag the data that will be imputed with NN
    imputed_flag = df_with_missing[[target]]
    imputed_flag[f'{target}_imputed_flag'] = np.where(
        imputed_flag[target].isnull(), 'neural network', None)
    imputed_flag.drop(target, axis=1, inplace=True)

    # For predicting drop the flags, because the neural network
    # doesn't take them
    baseline_imputed = baseline_imputed[
        [i for i in list(baseline_imputed.columns) if 'flag' not in i]]
    # For predicting we just need the rows where the target pollutant
    # is actually missing
    baseline_imputed = baseline_imputed[df_with_missing[target].isnull()]

    # Predict the target
    baseline_imputed = scaler.transform(baseline_imputed)
    predicted_target = model.predict(baseline_imputed)

    # Replace the missing values in the original dataframe with predicted ones
    index_of_missing = df_with_missing[target].isnull()
    data_with_imputed = df_with_missing.copy()
    data_with_imputed.loc[index_of_missing, target] = predicted_target

    # Add the flag to the predicted values
    data_with_imputed = data_with_imputed[
        metadata_columns + [target]
    ].join(imputed_flag).join(baseline_imputed_data_and_flags)

    # Rearrange the columns, so they are in a nicer order for
    # visual representation
    order_of_columns = metadata_columns + pollutants_list + [
        i + '_imputed_flag' for i in pollutants_list]
    data_with_imputed = data_with_imputed[order_of_columns]

    return data_with_imputed
