import base64
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import List

import folium
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from colour import Color
from ipywidgets import interact
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

FONT_SIZE_TICKS = 12
FONT_SIZE_TITLE = 20
FONT_SIZE_AXES = 16

# This is a list of categorical variables
categorical_variables = ['Station']
# This is a list of all pollutants
pollutants_list = ['PM2.5', 'PM10',  'NO', 'NO2', 'NOX', 'CO', 'OZONE']


def create_correlation_matrix(
    raw_data: pd.core.frame.DataFrame,
    features_list: List[str],
):
    """Creates a correlation matrix of the features.

    Args:
        raw_data (pd.core.frame.DataFrame): The data used.
        features_list (List[str]): List of features to include in the plot.

    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        raw_data[features_list].corr(),
        square=True,
        annot=True,
        cbar=False,
        cmap='RdBu',
        vmin=-1,
        vmax=1
    )
    plt.title('Correlation Matrix of Variables')

    plt.show()


def create_histogram_plot(df: pd.core.frame.DataFrame, bins: int):
    """Creates an interactive histogram.

    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.
        bins (int): number of bins for the histogram.

    """
    def _interactive_histogram_plot(station, pollutant):
        data = df[df.Station == station]
        x = data[pollutant].values
        try:
            plt.figure(figsize=(12, 6))
            plt.xlabel(f'{pollutant} concentration', fontsize=FONT_SIZE_AXES)
            plt.ylabel('Number of measurements', fontsize=FONT_SIZE_AXES)
            plt.hist(x, bins=bins)
            plt.title(
                f'Pollutant: {pollutant} - Station: {station}',
                fontsize=FONT_SIZE_TITLE)
            plt.xticks(fontsize=FONT_SIZE_TICKS)
            plt.yticks(fontsize=FONT_SIZE_TICKS)
            plt.show()
        except ValueError:
            print('Histogram cannot be shown '
                  'for selected values as there is no data')

    # Widget for picking the city
    station_selection = widgets.Dropdown(
        options=df.Station.unique(),
        description='Station'
    )

    # Widget for picking the continuous variable
    pollutant_selection = widgets.Dropdown(
        options=pollutants_list,
        description='Pollutant',
    )

    # Putting it all together
    interact(_interactive_histogram_plot, station=station_selection,
             pollutant=pollutant_selection)


def create_boxplot(df: pd.core.frame.DataFrame):
    """Creates a boxplot of pollutant values for each sensor station.

    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.

    """
    labels = df[categorical_variables[0]].unique()

    def _interactive_boxplot(cat_var):
        medians = []
        for value in df[categorical_variables[0]].unique():
            median = 1000
            try:
                rows = df[cat_var].loc[df[categorical_variables[0]] == value]
                if rows.isnull().sum() != rows.shape[0]:
                    median = rows.median()
            except Exception as e:
                print(f'Error processing {value}: {e}')
            medians.append(median)
        order_ind = np.argsort(medians)

        plt.figure(figsize=(17, 7))
        scale = 'linear'
        plt.yscale(scale)
        sns.boxplot(data=df, y=cat_var, x='Station',
                    order=labels[order_ind], color='seagreen')
        plt.title(f'Distributions of {cat_var}', fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Station', fontsize=FONT_SIZE_AXES)
        plt.ylabel(f'{cat_var} concentration', fontsize=FONT_SIZE_AXES)
        plt.xticks(fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()

    # Widget for picking the continuous variable
    cont_widget_histogram = widgets.Dropdown(
        options=pollutants_list,
        description='Pollutant',
    )

    interact(_interactive_boxplot, cat_var=cont_widget_histogram)


def create_scatterplot(df: pd.core.frame.DataFrame):
    """Creates a scatterplot for pollutant values. The pollutants on the x and
    y axis can be chosen with a dropdown menu.

    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.

    """
    df = df[pollutants_list]  # Take only the pollutants to scatter
    df_clean = df.dropna(inplace=False)

    def _interactive_scatterplot(var_x, var_y):
        x = df_clean[var_x].values
        y = df_clean[var_y].values
        bins = [200, 200]  # number of bins

        hh, locx, locy = np.histogram2d(x, y, bins=bins)
        z = np.array([hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])]
                     for a, b in zip(x, y)])
        idx = z.argsort()
        x2, y2, z2 = x[idx], y[idx], z[idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        s = ax.scatter(x2, y2, c=z2, cmap='jet', marker='.', s=1)

        ax.set_xlabel(f'{var_x} concentration', fontsize=FONT_SIZE_AXES)
        ax.set_ylabel(f'{var_y} concentration', fontsize=FONT_SIZE_AXES)

        ax.set_title(
            f'{var_x} vs. {var_y} (color indicates density of points)',
            fontsize=FONT_SIZE_TITLE
        )
        ax.tick_params(labelsize=FONT_SIZE_TICKS)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(
            s,
            cax=cax,
            cmap='jet',
            values=z2,
            orientation='vertical'
        )
        plt.show()
    cont_x_widget = widgets.Dropdown(
        options=pollutants_list,
        description='X-Axis'
    )
    cont_y_widget = widgets.Dropdown(
        options=pollutants_list,
        description='Y-Axis',
        value='PM10'
    )

    interact(_interactive_scatterplot,
             var_x=cont_x_widget, var_y=cont_y_widget)


def plot_pairplot(
    raw_data: pd.core.frame.DataFrame,
    features_list: List[str],
):
    """Creates a pairplot of the features.

    Args:
        raw_data (pd.core.frame.DataFrame): The data used.
        features_list (List[str]): List of features to include in the plot.

    """
    with sns.plotting_context(rc={'axes.labelsize': FONT_SIZE_AXES}):
        sns.pairplot(raw_data[features_list], kind='hist')
    plt.show()


def create_time_series_plot(df: pd.core.frame.DataFrame,
                            start_date: str, end_date: str):
    """Creates a time series plot, showing the concentration of pollutants over
    time. The pollutant and the measuring station can be selected with a
    dropdown menu.

    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.
        start_date (str): minimum date for plotting.
        end_date (str): maximum date for plotting.

    """
    def _interactive_time_series_plot(station, pollutant, date_range):
        data = df[df.Station == station]
        data = data[data.DateTime > date_range[0]]
        data = data[data.DateTime < date_range[1]]
        plt.figure(figsize=(12, 6))
        plt.plot(data['DateTime'], data[pollutant], '-')
        plt.title(f'Temporal change of {pollutant}', fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f'{pollutant} concentration', fontsize=FONT_SIZE_AXES)
        plt.xticks(rotation=20, fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()

    # Widget for picking the station
    station_selection = widgets.Dropdown(
        options=df.Station.unique(),
        description='Station'
    )

    # Widget for picking the pollutant
    pollutant_selection = widgets.Dropdown(
        options=pollutants_list,
        description='Pollutant',
    )

    dates = pd.date_range(start_date, end_date, freq='D')

    options = [(date.strftime(' %d/%m/%Y '), date) for date in dates]
    index = (0, len(options)-1)

    # Slider for picking the dates
    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '500px'}
    )

    # Putting it all together
    interact(_interactive_time_series_plot, station=station_selection,
             pollutant=pollutant_selection, date_range=selection_range_slider)


def add_extra_features(
        df: pd.core.frame.DataFrame,
        stations_csv_path: str
) -> pd.core.frame.DataFrame:
    """Adds new columns to the dataframe by joining it with another dataframe.

    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.
        stations_csv_path: str:

    Returns:
        df (pd.core.frame.DataFrame): The updated dataframe with new columns.

    """
    stations = pd.read_csv(stations_csv_path)
    stations = stations[['Sigla', 'Latitud', 'Longitud']]
    stations = stations.rename(
        columns={'Sigla': 'Station',
                 'Latitud': 'Latitude', 'Longitud': 'Longitude'})

    # This cell will convert the values in the columns
    # 'Latitud' and 'Longitud' to 'float64' (decimal) datatype
    stations['Latitude'] = stations['Latitude'].apply(parse_dms)
    stations['Longitude'] = stations['Longitude'].apply(parse_dms)

    df = pd.merge(df, stations, on='Station', how='inner')

    # This cell will extract information from the
    # 'datetime' column and generate months, day or week and hour columns
    df['day_of_week'] = pd.DatetimeIndex(df['DateTime']).day_name()
    df['hour_of_day'] = pd.DatetimeIndex(df['DateTime']).hour
    df.loc[df['hour_of_day'] == 0, 'hour_of_day'] = 24
    return df


def create_map_with_plots(
        full_data: pd.core.frame.DataFrame,
        image_dir: str,
        x_variable: str,
        y_variable: str = 'PM2.5'
) -> folium.Map:
    """Create a map to visualize geo points. The popup will show a scatterplot
    with the average daily/hourly emisions.

    Args:
        full_data (pd.core.frame.DataFrame): The dataframe with the data.
        image_dir:
        x_variable (str): The x variable on the popup plot.
        can be day_of_week or hour_of_day
        y_variable (str): A pollutant to be shown on y axis

    """

    data = full_data[['Latitude', 'Longitude',
                      y_variable, 'Station', x_variable]]
    data_grouped = data.groupby(['Station', x_variable]).agg(
        ({y_variable: ['mean', 'std']}))
    ymin = data_grouped[y_variable]['mean'].min()
    ymax = data_grouped[y_variable]['mean'].max()

    grouped_means = defaultdict(dict)
    grouped_stds = defaultdict(dict)
    for index, row in data_grouped.iterrows():
        grouped_means[index[0]][index[1]] = row[0]
        grouped_stds[index[0]][index[1]] = row[1]

    for key in grouped_means:
        if x_variable == 'day_of_week':
            keys = ['Sunday', 'Monday', 'Tuesday',
                    'Wednesday', 'Thursday', 'Friday', 'Saturday']
            label = 'daily average'
        else:
            keys = list(grouped_means[key].keys())
            label = 'hourly average'

        values = []
        stds = []
        for subkey in keys:
            values.append(grouped_means[key][subkey])
            stds.append(grouped_stds[key][subkey])
        values = np.array(values)
        stds = np.array(stds)
        plt.plot(keys, values, '-o', label=label)
        plt.fill_between(keys, values - stds, values + stds, alpha=0.2)
        if y_variable == 'PM2.5':
            plt.plot(keys, [12]*len(keys), '--g', label='recommended level')
        plt.plot(keys, [np.average(values)]*len(keys),
                 '--b', label='annual average')

        plt.ylim(ymin, ymax)
        plt.title(
            f'Station {key} avg. {y_variable} / {x_variable.split("_")[0]}')
        plt.ylabel(f'Avg. {y_variable} concentration')
        plt.xlabel(x_variable[0].upper() + x_variable[1:].replace('_', ' '))
        plt.legend(loc='upper left')
        if x_variable == 'day_of_week':
            plt.xticks(rotation=20)
        plt.savefig(f'{image_dir}/{key}.png')

        plt.clf()

    data_grouped_grid = data.groupby('Station').agg(
        ({y_variable: 'mean', 'Latitude': 'min', 'Longitude': 'min'}))

    data_grouped_grid_array = np.array(
        [
            data_grouped_grid['Latitude'].values,
            data_grouped_grid['Longitude'].values,
            data_grouped_grid[y_variable].values,
            data_grouped_grid.index.values
        ]
    ).T

    map3 = folium.Map(
        location=[data_grouped_grid_array[0]
                  [0], data_grouped_grid_array[0][1]],
        tiles='openstreetmap',
        zoom_start=11,
        width=1000,
        height=500
    )

    fg = folium.FeatureGroup(name='My Map')
    for lt, ln, pol, station in data_grouped_grid_array:
        image_path = f'{image_dir}/{station}.png'
        encoded_image = encode_image_to_base64(image_path)
        img_tag = f"<img src='data:image/png;base64,{encoded_image}'>"
        fg.add_child(folium.CircleMarker(
            location=[lt, ln],
            radius=15,
            popup=img_tag,
            fill_color=color_producer(y_variable, pol),
            color='',
            fill_opacity=0.5
        ))
        map3.add_child(fg)
    return map3


def plot_distribution_of_gaps(df: pd.core.frame.DataFrame, target: str):
    """Plots the distribution of the gap sizes in the dataframe

    Args:
        df (pd.core.frame.DataFrame): The dataframe
        target (str): The chosen pollutant for which it plots the distribution
    """

    def get_size_down_periods(data_frame, target_column):
        """Get the size of the downtime periods for the sensor"""
        gap_distribution = [0] * 4000
        x = []
        i = -1
        total_missing = 0
        count = 0
        for row in data_frame[target_column].values:
            if math.isnan(row):
                total_missing += 1
                if i == 0:
                    count = 1
                    i = 1
                else:
                    count += 1
            else:
                try:
                    if count > 0:
                        gap_distribution[count] += 1
                        x.append(count)
                except:
                    print(count)
                i = 0
                count = 0
        gap_distribution[0] = (
            data_frame[target_column].shape[0] - total_missing
        )
        return gap_distribution

    distribution = get_size_down_periods(df, target_column=target)
    for i in range(len(distribution)):
        distribution[i] = distribution[i] * i
    only_missing_per = distribution[1:-1]

    plt.figure(figsize=(10, 6))
    plt.plot(only_missing_per)
    plt.xlabel('Gap size (Hours)', fontsize=FONT_SIZE_AXES)
    plt.ylabel('Number of missing data points', fontsize=FONT_SIZE_AXES)
    plt.title('Distribution of gaps in the data', fontsize=FONT_SIZE_TITLE)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.show()


def visualize_missing_values_estimation(df: pd.core.frame.DataFrame,
                                        day: datetime):
    """Visualizes two ways of interpolating the data: nearest neighbor and
    last value and compares them to the real data

    Args:
        df (pd.core.frame.DataFrame): The dataframe
        day (datetime): The chosen day to plot
    """
    day = day.date()

    # Filter out the data for the day for the USM station
    rows_of_day = df.apply(lambda row: row['DateTime'].date() == day, axis=1)
    sample = df[rows_of_day]

    def draw(plot_data, station, missing_index, target):
        plot_data = plot_data.copy()
        plot_data.insert(
            0,
            'time_discriminator',
            (plot_data['DateTime'].dt.dayofyear * 100000 + plot_data[
                'DateTime'].dt.hour * 100).values,
            True
        )

        real = plot_data[plot_data['Station'] == station]
        example1 = real.copy()
        real = real.reset_index()
        example1 = example1.reset_index()
        example1.loc[missing_index, target] = float('NaN')

        missing = missing_index
        missing_before_after = [missing[0] - 1] + missing + [missing[-1] + 1]
        dates = set(list(example1.loc[missing_index, 'DateTime'].astype(str)))

        plt.figure(figsize=(10, 5))
        plt.plot(missing_before_after, real.loc[missing_before_after][target],
                 'r--o', label='actual values')

        plot_data_copy = plot_data.copy()
        plot_data_copy = plot_data_copy.reset_index()
        to_nan = plot_data_copy.apply(
            lambda row: str(row['DateTime']) in dates and row[
                'Station'] == station, axis=1)
        plot_data_copy.loc[to_nan, target] = float('NaN')
        imputer = KNNImputer(n_neighbors=1)
        imputer.fit(plot_data_copy[['time_discriminator', 'Latitude',
                                    'Longitude', target]])
        example1[f'new{target}'] = imputer.transform(
            example1[['time_discriminator', 'Latitude', 'Longitude', target]])[
                                   :, 3]
        plt.plot(missing_before_after,
                 example1.loc[missing_before_after][f'new{target}'], 'g--o',
                 label='nearest neighbor')

        plt.plot(example1.index, example1[target], '-*')

        example1[f'ffill{target}'] = example1.fillna(method='ffill')[target]
        plt.plot(missing_before_after,
                 example1.loc[missing_before_after][f'ffill{target}'], 'y--*',
                 label='last known value')

        plt.xlabel('Hour of day', fontsize=FONT_SIZE_AXES)
        plt.ylabel(f'{target} concentration', fontsize=FONT_SIZE_AXES)
        plt.title('Estimating missing values', fontsize=FONT_SIZE_TITLE)
        plt.legend(loc="upper left", fontsize=FONT_SIZE_TICKS)
        plt.xticks(fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        plt.show()

    def selector(station, hour_start, window_size, target):
        missing_index_list = list(range(hour_start, hour_start + window_size))
        draw(
            plot_data=sample,
            station=station,
            missing_index=missing_index_list,
            target=target
        )
    # Widgets for selecting the parameters
    station_selection = widgets.Dropdown(
        options=df.Station.unique(),
        description='Station',
        value='USM'
    )
    target_pollutant_selection = widgets.Dropdown(
        options=pollutants_list,
        description='Pollutant',
        value='PM2.5'
    )
    hour_start_selection = widgets.Dropdown(
        options=list([2, 3, 4, 5, 6, 7, 8, 9, 10]),
        description='Hour start',
        value=3
    )
    window_size_selection = widgets.Dropdown(
        options=list([1, 2, 3, 5, 6, 9, 12]),
        description='Window size',
        value=1
    )

    return interact(
        selector,
        station=station_selection,
        hour_start=hour_start_selection,
        window_size=window_size_selection,
        target=target_pollutant_selection
    )


def create_plot_with_preditions(
        df: pd.core.frame.DataFrame,
        model: tf.keras.Model,
        scaler: StandardScaler,
        feature_names: List[str],
        target: str,
        start_date: datetime,
        end_date: datetime
):
    """
    This function will take the features (x), the target (y) and the model
    and will fit and Evaluate the model.

    Args:
        df (pd.core.frame.DataFrame): The dataframe with the data.
        model (tf.keras.Model): Model
        scaler (StandardScaler): scaler
        feature_names (List[str]): Names of feature columns
        target (str): Name of the target column
        start_date (str): minimum date for plotting.
        end_date (str): maximum date for plotting.
    """

    def draw_example3(sample, station, predicted2, missing_index):
        sample = sample.copy()
        sample.insert(0, 'time_discriminator',
                      (sample['DateTime'].dt.dayofyear * 100000 + sample[
                          'DateTime'].dt.hour * 100).values, True)

        real_data = sample[sample['Station'] == station]
        example1 = real_data.copy()
        real_data = real_data.reset_index()
        example1 = example1.reset_index()
        example1.loc[missing_index, target] = float('NaN')

        missing = missing_index
        missing_before_after = [missing[0] - 1] + missing + [missing[-1] + 1]
        dates = set(list(example1.loc[missing_index, 'DateTime'].astype(str)))

        plt.plot(missing_before_after,
                 real_data.loc[missing_before_after][target], 'r--o',
                 label='actual values')

        copy_of_data = sample.copy()
        copy_of_data = copy_of_data.reset_index()
        to_nan = copy_of_data.apply(
            lambda row: str(row['DateTime']) in dates and row[
                'Station'] == station, axis=1)
        copy_of_data.loc[to_nan, target] = float('NaN')

        imputer = KNNImputer(n_neighbors=1)
        imputer.fit(copy_of_data[
                        ['time_discriminator', 'Latitude', 'Longitude',
                         target]])
        example1[f'new_{target}'] = imputer.transform(
            example1[['time_discriminator', 'Latitude', 'Longitude', target]])[
                                    :, 3]

        plt.plot(missing_before_after,
                 example1.loc[missing_before_after][f'new_{target}'], 'g--o',
                 label='nearest neighbor')
        plt.plot(example1.index, example1[target], '-*')

        example1[f'nn_{target}'] = example1[target].copy()
        example1.loc[missing, f'nn_{target}'] = predicted2[np.array(missing)]
        plt.plot(missing_before_after,
                 example1.loc[missing_before_after][f'nn_{target}'], 'y--*',
                 label='neural network')

        plt.xlabel('Index', fontsize=FONT_SIZE_AXES)
        plt.ylabel(f'{target} concentration', fontsize=FONT_SIZE_AXES)
        plt.title('2 days data and predictions', fontsize=FONT_SIZE_TITLE)
        plt.legend(loc="upper left", fontsize=FONT_SIZE_TICKS)
        plt.xticks(fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)

    def plot_predictions(station, size, start_index):
        try:
            data = df[df.DateTime > start_date]
            data = data[data.DateTime < end_date]

            x_test = data[df.Station == station]
            x_test = x_test[feature_names]
            x_test = scaler.transform(x_test)
            # y_test = data[target]

            y_predicted = model.predict(x_test)

            plt.figure(figsize=(10, 5))
            draw_example3(data, station, y_predicted,
                          list(range(start_index, start_index + size)))
            plt.show()
        except Exception as e:
            print(
                'The selected range cannot be plotted due to missing values. '
                'Please select other values.\n')
            print(e)

    # Widget for picking the station
    station_selection = widgets.Dropdown(
        options=df.Station.unique(),
        description='Station'
    )
    # Widget for picking the window size
    windows_size_selection = widgets.Dropdown(
        options=list([1, 2, 3, 5, 6, 12, 24]),
        description='Window'
    )
    # Widget for selecting index of data
    index_selector = widgets.IntSlider(value=1,
                                       min=1,
                                       max=48,
                                       step=1,
                                       description='Index')

    interact(plot_predictions, station=station_selection,
             size=windows_size_selection, start_index=index_selector)


# The functions from here are helper functions,
# that are used by other functions.
# These functions are not directly called in the notebook.
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def parse_dms(coor: str) -> float:
    """Transforms strings of degrees, minutes and seconds to a decimal value.

    Args:
        coor (str): String containing degrees in DMS format

    Returns:
        dec_coord (float): coordinates as a decimal value

    """
    parts = re.split(r'[^\d\w]+', coor)
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2]+'.'+parts[3])
    direction = parts[4]

    dec_coord = degrees + minutes / 60 + seconds / 3600

    if direction == 'S' or direction == 'W':
        dec_coord *= -1

    return dec_coord


def color_producer(pollutant_type, pollutant_value):
    """This function returns colors based on the pollutant values to create a
    color representation of air pollution.

    The color scale  for PM2.5 is taken from purpleair.com
    and it agrees with international guidelines
    The scale for other pollutants was created based
    on the limits for other pollutants to approximately
    correspond to the PM2.5 color scale. The values in the scale should not
    be taken for granted and
    are used just for the visualization purposes.

    Args:
        pollutant_type (str): Type of pollutant to get the color for
        pollutant_value (float): Value of polutant concentration

    Returns:
        pin_color (str): The color of the bucket

    """
    all_colors_dict = {
        'PM2.5': {
            0: 'green', 12: 'yellow', 35: 'orange', 55.4: 'red', 150: 'black'},
        'PM10': {
            0: 'green', 20: 'yellow', 60: 'orange', 110: 'red', 250: 'black'},
        'CO': {
            0: 'green', 4: 'yellow', 10: 'orange', 20: 'red', 50: 'black'},
        'OZONE': {
            0: 'green', 60: 'yellow', 100: 'orange', 200: 'red', 300: 'black'},
        'NOX': {
            0: 'green', 40: 'yellow', 80: 'orange', 160: 'red', 300: 'black'},
        'NO': {
            0: 'green', 40: 'yellow', 80: 'orange', 160: 'red', 300: 'black'},
        'NO2': {
            0: 'green', 20: 'yellow', 40: 'orange', 80: 'red', 200: 'black'},
    }

    # Select the correct color scale, if it is not available, choose PM2.5
    colors_dict = all_colors_dict.get(pollutant_type, all_colors_dict['PM2.5'])
    thresholds = sorted(list(colors_dict.keys()))

    previous = 0
    for threshold in thresholds:
        if pollutant_value < threshold:
            bucket_size = threshold - previous
            bucket = (pollutant_value - previous) / bucket_size
            colors = list(Color(colors_dict[previous]).range_to(
                Color(colors_dict[threshold]), 11))
            pin_color = str(colors[int(np.round(bucket*10))])
            return pin_color
        previous = threshold
