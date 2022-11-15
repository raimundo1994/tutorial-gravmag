def WGS84():
    '''
    This function returns the following parameters defining the
    reference elipsoid WGS84:
    a = semimajor axis [m]
    f = flattening

    output:
    a, f
    '''
    a = 6378137.0
    f = 1/298.257223563

    return a, f

def SAD69():
    '''
    This function returns the following parameters defining the
    reference elipsoid SAD69:
    a = semimajor axis [m]
    f = flattening

    output:
    a, f
    '''
    a = 6378160.0
    f = 1.0/298.25

    return a, f

def Hayford1924():
    '''
    This function returns the following parameters defining the
    International (Hayford's) reference elipsoid of 1924:
    a = semimajor axis [m]
    f = flattening

    output:
    a, f
    '''
    a = 6378388.0
    f = 1.0/297.0

    return a, f

def SAD69_WGS84():
    '''
    Transformation parameters from local geodetic system
    SAD69 to WGS84.

    output
    dx: float - origin translation along the x-axis (in meters).
    dy: float - origin translation along the y-axis (in meters).
    dz: float - origin translation along the z-axis (in meters).
    '''

    dx = -57.
    dy = 1.
    dz = -41.

    return dx, dy, dz
