import numpy as np
import pandas as pd

def extract_X_Y_Z(file):
    '''
    Extract the IGRF geocentric components in the northward, eastward, and
    radially inwards directions (X, Y and Z), all in nT, from the output file
    generated with the NGDC software "Geomag" version 7.0
    (https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html).


    Parameters:

    file : string
        Path to the output file.

    Returns:

    X, Y, Z: arrays
        X, Y and Z components (in nT) of the computed IGRF.
    '''
    column_names = ['Date', 'Coord-System', 'Altitude', 'Latitude', 'Longitude',
                'D_deg', 'D_min', 'I_deg', 'I_min',
                'H_nT', 'X_nT', 'Y_nT', 'Z_nT', 'F_nT',
                'dD_min', 'dI_min', 'dH_nT', 'dX_nT', 'dY_nT', 'dZ_nT', 'dF_nT']

    column_types = ['object', 'object', 'object', 'float64', 'float64',
                    'object', 'object', 'object', 'object',
                    'float64', 'float64', 'float64', 'float64', 'float64',
                    'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64']

    column_types_dict = dict()

    for cn, ct in zip(column_names, column_types):
        column_types_dict[cn] = ct

    igrf = pd.read_csv(file, sep='\s+', header=0,
                       names=column_names, dtype=column_types_dict,
                       usecols=['X_nT', 'Y_nT', 'Z_nT'])

    X = igrf['X_nT']
    Y = igrf['Y_nT']
    Z = igrf['Z_nT']

    return X.values, Y.values, Z.values

def extract_D_I_F(file):
    '''
    Extract the IGRF declination, inclination (both in decimal degrees) and
    intensity (in nT) from the outfiles generated with the NGDC software
    "Geomag" version 7.0 (https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html).

    Parameters:

    file : string
        Path to the output file.

    Returns:

    declination, inclination, intensity: arrays
        Vectors containing the declination, inclination (both in decimal
        degrees) and intensity (in nT) of the computed IGRF.
    '''
    column_names = ['Date', 'Coord-System', 'Altitude', 'Latitude', 'Longitude',
                'D_deg', 'D_min', 'I_deg', 'I_min',
                'H_nT', 'X_nT', 'Y_nT', 'Z_nT', 'F_nT',
                'dD_min', 'dI_min', 'dH_nT', 'dX_nT', 'dY_nT', 'dZ_nT', 'dF_nT']

    column_types = ['object', 'object', 'object', 'float64', 'float64',
                    'int64', 'int64', 'int64', 'int64', 'float64', 'float64',
                    'float64', 'float64', 'float64', 'float64', 'float64',
                    'float64', 'float64', 'float64', 'float64', 'float64']

    column_types_dict = dict()

    for cn, ct in zip(column_names, column_types):
        column_types_dict[cn] = ct

    igrf = pd.read_csv(file, sep='\s+', header=0,
                       names=column_names, dtype=column_types_dict,
                       usecols=['D_deg', 'D_min', 'I_deg', 'I_min', 'F_nT'])

    declination = igrf['D_deg'] + \
                  np.sign(igrf['D_deg'])*(igrf['D_min']/60)

    inclination = igrf['I_deg'] + \
                  np.sign(igrf['I_deg'])*(igrf['I_min']/60)

    intensity = igrf['F_nT']

    return declination.values, inclination.values, intensity.values
