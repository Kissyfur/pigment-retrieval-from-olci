import numpy as np
import re
import datetime
import xarray as xr

def is_integer(input_str):
    try:
        int(input_str)
        return True
    except Exception as e:
        return False


def is_number(input_str):
    try:
        float(input_str)
        return True
    except Exception as e:
        return False

def replace_special_characters(input_string):
    # Replace all non-alphanumeric characters (except underscore) with underscore
    return re.sub(r'[^a-zA-Z0-9_]', '_', input_string)


def rename_wavelengths(ds):
    wavelengths = [400, 412, 442, 490, 510, 560, 620, 665, 673, 681, 708, 778, 865]
    current_wavelengths = ds.wavelength.values
    for wl in wavelengths:
        closer_wl_ind = np.argmin(np.abs(current_wavelengths-wl))
        current_wavelengths[closer_wl_ind] = wl
    new_ds = ds.assign_coords({'wavelength': current_wavelengths})
    return new_ds.sel(wavelength=wavelengths)


def TRS_common_read_Q(fname):
    d = {}
    data = []
    read = False
    with open(fname, 'r') as file:
        for line in file:
            stringa = line.strip('/\n')
            if stringa == 'end_data':
                read = False
            if read:
                stringa = stringa.strip().split(',')[1:]
                data.append(stringa)
            if stringa == 'begin_data':
                read = True

            stringa = line.strip('/\n').split('=')
            if len(stringa) == 2:
                valid_field = replace_special_characters(stringa[0])
                if valid_field == 'records' or valid_field == 'fields':
                    measures = stringa[1].split(',')
                    measures = [replace_special_characters(mes) if '/' in mes else mes for mes in measures ]
                else:
                    d[valid_field] = stringa[1]
                if is_number(stringa[1]):
                    d[valid_field] = float(stringa[1])
                if is_integer(stringa[1]):
                    d[valid_field] = int(stringa[1])
                if valid_field == 'sampling_date':
                    d[valid_field] = datetime.datetime.strptime(stringa[1], "%d/%m/%Y")
    data_z = np.empty((d['n_records'], d['n_fields']))
    data_z[:] = np.nan
    for i, zer in enumerate(data_z):
        data_line = data[i]
        zer[0:len(data_line)] = data_line

    arr = data_z[1:]
    var_names = measures[1:]
    data_vars = {key: (["wavelength"], val) for key, val in zip(var_names, arr)}
    data_vars = data_vars | d # {key: (["point"], [val]) for key, val in d.items()}

    c = {"wavelength": data_z[0].astype(int)}
    xar = xr.Dataset(data_vars=data_vars, coords=c)

    if 'avg_RRS0_from_Lu0_fQ' in xar:
        xar = xar.rename({'avg_RRS0_from_Lu0_fQ': 'rrs'})
    elif 'avg_RRS_0_0[sr^-1]' in xar:
        xar = xar.rename({'avg_RRS_0_0[sr^-1]': 'rrs'})
    elif 'RRS00_M02[sr^-1]' in xar:
        xar = xar.rename({'RRS00_M02[sr^-1]': 'rrs'})
        xar['rrs'] = xar['rrs'] / 100
    xar = xar.rename({'latitude_deg_': 'lat', 'longitude_deg_': 'lon'})
    return xar


def TRS_common_read_H(fname):
    d = {}
    data = []
    read = False
    with open(fname, 'r') as file:
        for line in file:
            stringa = line.strip('/\n')
            if stringa == 'end_data':
                read = False
            if read:
                stringa = stringa.strip().split(',')
                stringa = [float(s) for s in stringa]
                data.append(stringa)
            if stringa == 'begin_data':
                read = True

            stringa = line.strip('/\n').split('=')
            if len(stringa) == 2:
                valid_field = replace_special_characters(stringa[0])
                if valid_field == 'records' or valid_field == 'fields':
                    measures = stringa[1].split(',')
                else:
                    d[valid_field] = stringa[1]
                if is_number(stringa[1]):
                    d[valid_field] = float(stringa[1])
                if is_integer(stringa[1]):
                    d[valid_field] = int(stringa[1])
                if stringa[0] == 'sampling_date':
                    d[valid_field] = datetime.datetime.strptime(stringa[1], "%d/%m/%Y")

    # return [ d|dict(zip(measures, data_line)) for data_line in data if data_line[1] != -9.99]
    return [d|dict(zip(measures, data_line)) for data_line in data if data_line[1] == 0]

matchups_TRS = ["matchup_TRS2", "matchup_TRS6", "matchup_TRS1", "matchup_TRS3",
                "eastern_med"
                ]