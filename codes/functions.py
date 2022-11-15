import numpy as np
from . import prism as p
from numba import njit

def build_model(x,y,z,dz):

    '''
    Compute a prism mesh model containing west, east, south, north, top, and bottom boundaries.

    Parameters:
    --------
    x: numpy array 2D
        Matrix containing the x component of TCC (in meters).
    y: numpy array 2D
        Matrix containing the y component of TCC (in meters).
    z: numpy array 2D
        Matrix containing the z component of TCC (in meters).
    dz: scalar
        Value added to z component matrix (in meters)

    Returns:
    --------
    result: numpy array 2D
            Matrix containing the coordinates of the boundaries a grid prisms.
            the following order: west (y1), east (y2), south (x1), north (x2), top (z1) and bottom (z2)

    '''

    assert x.shape == y.shape == z.shape, 'x, y and z must be the same size.'
    assert np.isscalar(dz) and dz>0,'dz must be a positive scalar. '

    shape = x.shape

    dy = y[0,1:] - y[0,:-1]
    dx = x[:-1,0]- x[1:,0]

    #west boundary
    y1 = np.zeros_like(y[0,:])
    y1[1:] = y[0,1:] - dy/2
    y1[0] = y[0,0] - dy[0]/2
    #east boundary
    y2 = np.zeros_like(y[0,:])
    y2[:-1] = y[0,:-1] + dy/2
    y2[-1] = y[0,-1] + dy[-1]/2
    #south boundary
    x1 = np.zeros_like(x[:,0])
    x1[1:] = x[1:,0] - dx/2
    x1[0] = x[0,0] - dx[0]/2
    #north boundary
    x2 = np.zeros_like(x[:,0])
    x2[:-1] = x[:-1,0] + dx/2
    x2[-1] = x[-1,0] + dx[-1]/2

    y1 = np.resize(y1,shape)
    y2 = np.resize(y2,shape)
    x1 = np.resize(x1,shape[::-1]).T
    x2 = np.resize(x2,shape[::-1]).T

    west = y1.ravel()
    east = y2.ravel()
    south = x1.ravel()
    north = x2.ravel()
    top = z.ravel()
    bottom = z.ravel()+dz

    model = [west,east,south,north,top,bottom]

    model = np.array(model).T

    return model

def grid_point(area,shape):

    '''
    Create a regular grid. The latitude is North-South and longitude East-West.

    Parameters:
    --------
    area [x1, x2, y1, y2]: List containing the borders of the grid.
    shape [nx, ny]: List containing the shape of the regular grid.

    Returns:
    --------
    [longitude, latitude]: numpy array 2D
                           Numpy arrays with the Latitude(axis x) and Longitude(axis y)
                           coordinates of the grid points.

    '''

    nx, ny = shape
    x1, x2, y1, y2 = area

    assert x1 <= x2, \
        "Invalid area dimensions {}, {}. x1 must be < x2.".format(x1, x2)
    assert y1 <= y2, \
        "Invalid area dimensions {}, {}. y1 must be < y2.".format(y1, y2)

    longitude = np.linspace(x1, x2, ny)
    latitude = np.linspace(y2, y1, nx)

    longitude,latitude = np.meshgrid(longitude,latitude)

    return longitude,latitude



def sensitivity_matrix(coordinates, model, magnetization, inclination, declination):

    '''
    Compute the sensitivity matrix

    Parameters:
    --------
    coordinates : numpy array 2D
        Containing y (first line), x (second line), and z (third line) of
        the computation points. All coordinates should be in meters.

    model : numpy array 2D
        Containing the coordinates of the prisms. Each line must contain
        the coordinates of a single prism in the following order:
        west (y1), east (y2), south (x1), north (x2), top (z1) and bottom (z2).
        All coordinates should be in meters.

    magnetization : numpy array 2D
        Containing the total-magnetization components of the prisms.
        Each line must contain the intensity (in A/m), inclination and
        declination (in degrees) of the total magnetization of a single prism.

    inclination : numpy array 1D
         Containing inclination (in degrees) in TCC of the observation points.

    declination : numpy array 1D
         Containing  declination (in degrees) in TCC of the observation points.

    Returns:
    --------

    result: numpy array 2D
            Sensitivity matrix N x M,
            where N is observations number and M parameters number.

    '''

    assert model.ndim == 2, 'model must be a 2D'
    assert magnetization.ndim == 2,'magnetization must be a 2D'
    assert model.shape[1] == 6,'model must have 6 columns'
    assert magnetization.shape[1] == 3, 'magnetization must have 3 columns'
    assert coordinates.ndim == 2, 'coordinates must be a 2D'
    assert coordinates.shape[0] == 3, 'coordinates must have 3 rows'


    N = coordinates.shape[1]
    M = model.shape[0]

    A = np.zeros((N,M), dtype= np.float64)

    cosI0 = np.cos(np.deg2rad(inclination))
    sinI0 = np.sin(np.deg2rad(inclination))
    cosD0 = np.cos(np.deg2rad(declination))
    sinD0 = np.sin(np.deg2rad(declination))
    Fx = cosI0*cosD0
    Fy = cosI0*sinD0
    Fz = sinI0

    for j,(m,mag) in enumerate(zip(model, magnetization)):
        m = np.array([m])
        mag = np.array([mag])
        bx = p.mag(coordinates=coordinates, prisms=m, magnetization=mag, field='b_x')
        by = p.mag(coordinates=coordinates, prisms=m, magnetization=mag, field='b_y')
        bz = p.mag(coordinates=coordinates, prisms=m, magnetization=mag, field='b_z')

        A[:,j] = Fx*bx + Fy*by + Fz*bz

    return A


def statistical(data, unit = '(No Unit)'):  
    '''
    A statistical function that calculates the minimum and maximum values for a simple 
    dataset and also its mean values and variations. The dataset can be a 1D or a 2D
    numpy array.
    
    Parameters:
    --------
    data - numpy array - data set as a vector
    unit - string - data unit
    
    Returns:
    --------
    datamin - float - minimun value for the data
    datamax - float - maximun value for the data
    datamed - float - mean value for all dataset
    datavar - float - variation for all dataset    
    '''
    
    assert data.size > 1, 'Data set must have more than one element!'
    
    datamin = np.min(data)
    datamax = np.max(data)
    datamed = np.mean(data)
    datavar = datamax - datamin
    
    print ('Minimum:    %5.4f' % (datamin), unit)
    print ('Maximum:    %5.4f' % (datamax), unit)
    print ('Mean value: %5.4f' % (datamed), unit)
    print ('Variation:  %5.4f' % (datavar), unit)
    # Return the final analysis    
    return datamin, datamax, datamed, datavar

