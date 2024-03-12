import json
from typing import Tuple

import numpy as np
from shapely.geometry import Point, Polygon, shape
from sklearn.neighbors import KNeighborsRegressor


def predict_on_bogota(
        model: KNeighborsRegressor,
        n_points: int = 64
) -> Tuple[np.ndarray, float, float]:
    """Creates a grid of predicted pollutant values based on the neighboring
    stations.

    Args:
        model (KNeighborsRegressor): Model to use
        n_points (int): number of points in the grid

    Returns:
        predictions_xy (np.ndarray): array containing tuples of coordinates
        and predicted value

        dlat (float): latitude size of grid
        dlon (float): longitudinal size of grid

    """
    with open('data/bogota.json') as f:
        js = json.load(f)

    # Check each polygon to see if it contains the point
    polygon = Polygon(shape(js['features'][0]['geometry']))
    (lon_min, lat_min, lon_max, lat_max) = polygon.bounds

    dlat = (lat_max - lat_min) / (n_points - 1)
    dlon = (lon_max - lon_min) / (n_points - 1)
    lat_values = np.linspace(lat_min - dlat, lat_max + dlat, n_points)
    lon_values = np.linspace(lon_min - dlon, lon_max + dlon, n_points)
    xv, yv = np.meshgrid(lat_values, lon_values, indexing='xy')

    predictions_xy = []

    for i in range(n_points):
        # row = [0] * n_points
        for j in range(n_points):
            if polygon.contains(Point(lon_values[j], lat_values[i])):
                point = [lat_values[i], lon_values[j]]
                # Remove the data of the same station
                pred = model.predict([point])
                predictions_xy.append(
                    [lat_values[i], lon_values[j], pred[0][0]])

    predictions_xy = np.array(predictions_xy)

    return predictions_xy, dlat, dlon
