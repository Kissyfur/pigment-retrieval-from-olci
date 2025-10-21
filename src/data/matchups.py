import pandas as pd
import xarray as xr
import re
import numpy as np
import itertools as it

from tqdm import tqdm

LAT, LON, TIME = 'lat', 'lon', 'time'

# Thresholds for mathcup
TIME_TH = pd.Timedelta(days=1)
LAT_TH = 0.1
LON_TH = 0.1

# Matchup region extension (window)
TIME_WINDOW = pd.Timedelta(days=1)
LAT_WINDOW = 0.06
LON_WINDOW = 0.06


# Select data with enough values (discard when majority is nan)

def dimension_len(ds, dim_name):
    if dim_name not in ds.sizes.keys():
        return 1
    return ds.sizes[dim_name]


def select_valid_data(ds, threshold=0.2):
    cube_dimension = (dimension_len(ds, LAT) * dimension_len(ds, LON) *
                      dimension_len(ds, TIME))
    values_count = (~ds.isnull()).sum(dim=(LAT, LON))
    if 'time' in ds.dims:
        values_count = (~ds.isnull()).sum(dim=(LAT, LON, TIME))
    no_null_percent = values_count / cube_dimension
    valid_indices = (no_null_percent > threshold).to_array().prod(axis=0) != 0
    return ds.isel(Id=valid_indices)


def manhattan_distance(arr, p2):
    return np.sum(np.abs(arr-p2), axis=-1)


def radius_weights(sh):
    center = np.array(sh) // 2
    coords = it.product(*[list(range(i)) for i in sh])
    coords = np.array(list(coords))
    dist = manhattan_distance(coords, center)
    res = dist.reshape(sh)
    return res
# conv_sort = ["time", "lat", "lon"]


def compute_weights(ds):
    ds_cp = ds.copy()
    weights_arr = radius_weights(ds_cp.values.shape)
    conv_sort = list(ds_cp.dims)
    max_w = np.max(weights_arr)
    weights_arr = -weights_arr + max_w
    weights_arr[1] = weights_arr[1] * 3
    weights_arr = np.exp(weights_arr)

    weights = xr.DataArray(data=weights_arr, dims=conv_sort)
    return weights


def average(ds):
    coords_lat = ds.coords['lat'].mean(axis=1)
    coords_lon = ds.coords['lon'].mean(axis=1)
    if 'time' in ds.dims:
        coords_time = ds.coords['time'].mean(axis=1)
        ds_mean = ds.mean(dim=['lat', 'lon', 'time'], skipna=True, keep_attrs=True)
        ds_mean = xr.merge([ds_mean, coords_lat, coords_lon, coords_time])
    else:
        ds_mean = ds.mean(dim=['lat', 'lon'], skipna=True, keep_attrs=True)
        ds_mean = xr.merge([ds_mean, coords_lat, coords_lon])
    return ds_mean


def radius_weighted_average(ds):
    first_variable = list(ds.keys())[0]
    weights = compute_weights(ds.isel(Id=0)[first_variable])
    coords_lat = ds.coords['lat'].mean(axis=1)
    coords_lon = ds.coords['lon'].mean(axis=1)
    if 'time' in weights.dims:
        coords_time = ds.coords['time'].mean(axis=1)
        ds_weighted = ds.weighted(weights).mean(dim=['lat', 'lon', 'time'], skipna=True, keep_attrs=True)
        ds_weighted = xr.merge([ds_weighted, coords_lat, coords_lon, coords_time])
    else:
        ds_weighted = ds.weighted(weights).mean(dim=['lat', 'lon'], skipna=True, keep_attrs=True)
        ds_weighted = xr.merge([ds_weighted, coords_lat, coords_lon])
    return ds_weighted


def dms_to_decimal(dms):
    # Return as-is if already a number or NaN
    if pd.isna(dms) or isinstance(dms, (int, float)):
        return dms

    # Normalize all weird symbols and spacing
    dms = (str(dms)
           .replace("º", "°").replace("° ", "°")
           .replace("’", "'").replace("‘", "'")
           .replace("′", "'")
           .replace("″", "\"").replace("”", "\"").replace("“", "\"")
           .replace("''", "\"").replace("  ", " ")
           .strip())

    # Match pattern: degrees, minutes, seconds, direction
    match = re.match(r"(\d+)°\s*(\d+)'?\s*([\d\.]+)?\"?\s*([NSEW])?", dms)
    if not match:
        return dms  # leave unchanged if format unexpected

    deg, minutes, seconds, direction = match.groups()

    # Default seconds to 0 if missing
    seconds = seconds or 0

    # Convert to decimal degrees
    dec = float(deg) + float(minutes)/60 + float(seconds)/3600

    # Flip sign for South or West
    if direction in ['S', 'W']:
        dec = -dec

    return dec


def match_up(ids, lats, lons, times, region, lat_win=LAT_WINDOW, lon_win=LON_WINDOW, time_win=TIME_WINDOW):
    match_ups = []

    for id_, lat, lon, date in tqdm(zip(ids, lats, lons, times), total=len(ids)):
        near_point = region.sel({TIME: date, LAT: lat, LON: lon}, method='nearest')

        near_date, near_lat, near_lon = near_point[TIME].values, near_point[LAT].values, near_point[LON].values
        if abs(near_date - date) < TIME_TH and abs(near_lon - lon) < LON_TH and abs(near_lat - lat) < LAT_TH:
            match = region.sel({TIME: slice(near_date - time_win, near_date + time_win),
                                    LAT: slice(near_lat - lat_win, near_lat + lat_win),
                                    LON: slice(near_lon - lon_win, near_lon + lon_win)})
            lats_coord = xr.DataArray([match[LAT].values], dims=['Id', LAT])
            lons_coord = xr.DataArray([match[LON].values], dims=['Id', LON])
            time_coord = xr.DataArray([match[TIME].values], dims=['Id', TIME])
            if 'depth' in list(match.coords):
                match = match.sel(depth=region.depth.min())
            match = match.assign_coords({LON: lons_coord, LAT: lats_coord, TIME: time_coord, 'Id': ('Id', [id_])})
            match_ups.append(match)
    return match_ups