import warnings
import numpy

import numpy
import scipy.interpolate
import math

from matplotlib import pyplot, widgets
# Quick hack so that the docs can build using the mocks for readthedocs
# Ideal would be to log an error message saying that functions from pyplot
# were not imported
try:
    from matplotlib.pyplot import *
except:
    pass



def contourf(x, y, v, shape, levels, interp=False, extrapolate=False,
             vmin=None, vmax=None, cmap=pyplot.cm.jet, basemap=None):
    """
    Make a filled contour plot of the data.

    Parameters:

    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * levels : int or list
        Number of contours to use or a list with the contour values.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
    * vmin, vmax
        Saturation values of the colorbar. If provided, will overwrite what is
        set by *levels*.
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`~fatiando.vis.mpl.basemap` for creating basemaps)

    Returns:

    * levels : list
        List with the values of the contour levels

    """
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if interp:
        x, y, v = interp(x, y, v, shape, extrapolate == extrapolate)

    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap, picker=True)
    if basemap is None:
        ct_data = pyplot.contourf(X, Y, V, levels, **kwargs)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contourf(lon, lat, V, levels, **kwargs)
    return ct_data.levels



def interp(x, y, v, shape, area=None, algorithm='cubic', extrapolate=False):
    """
    Interpolate data onto a regular grid.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * shape : tuple = (nx, ny)
        Shape of the interpolated regular grid, ie (nx, ny).
    * area : tuple = (x1, x2, y1, y2)
        The are where the data will be interpolated. If None, then will get the
        area from *x* and *y*.
    * algorithm : string
        Interpolation algorithm. Either ``'cubic'``, ``'nearest'``,
        ``'linear'`` (see scipy.interpolate.griddata).
    * extrapolate : True or False
        If True, will extrapolate values outside of the convex hull of the data
        points.

    Returns:

    * ``[x, y, v]``
        Three 1D arrays with the interpolated x, y, and v

    """
    if algorithm not in ['cubic', 'linear', 'nearest']:
        raise ValueError("Invalid interpolation algorithm: " + str(algorithm))
    nx, ny = shape
    if area is None:
        area = (x.min(), x.max(), y.min(), y.max())
    x1, x2, y1, y2 = area
    xp, yp = regular(area, shape)
    grid = interp_at(x, y, v, xp, yp, algorithm==algorithm,
                     extrapolate==extrapolate)
    return [xp, yp, grid]

def interp_at(x, y, v, xp, yp, algorithm='cubic', extrapolate=False):
    """
    Interpolate data onto the specified points.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * xp, yp : 1D arrays
        Points where the data values will be interpolated
    * algorithm : string
        Interpolation algorithm. Either ``'cubic'``, ``'nearest'``,
        ``'linear'`` (see scipy.interpolate.griddata)
    * extrapolate : True or False
        If True, will extrapolate values outside of the convex hull of the data
        points.

    Returns:

    * v : 1D array
        1D array with the interpolated v values.

    """
    if algorithm not in ['cubic', 'linear', 'nearest']:
        raise ValueError("Invalid interpolation algorithm: " + str(algorithm))
    grid = scipy.interpolate.griddata((x, y), v, (xp, yp),
                                      method=algorithm).ravel()
    if extrapolate and algorithm != 'nearest' and numpy.any(numpy.isnan(grid)):
        grid = extrapolate_nans(xp, yp, grid)
    return grid

def extrapolate_nans(x, y, v):
    """
    Extrapolate the NaNs or masked values in a grid INPLACE using nearest
    value.

    .. warning:: Replaces the NaN or masked values of the original array!

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.

    Returns:

    * v : 1D array
        The array with NaNs or masked values extrapolated.

    """
    if numpy.ma.is_masked(v):
        nans = v.mask
    else:
        nans = numpy.isnan(v)
    notnans = numpy.logical_not(nans)
    v[nans] = scipy.interpolate.griddata((x[notnans], y[notnans]), v[notnans],
                                         (x[nans], y[nans]),
                                         method='nearest').ravel()
    return v


def regular(area, shape, z=None):
    """
    Create a regular grid.

    The x directions is North-South and y East-West. Imagine the grid as a
    matrix with x varying in the lines and y in columns.

    Returned arrays will be flattened to 1D with ``numpy.ravel``.

    .. warning::

        As of version 0.4, the ``shape`` argument was corrected to be
        ``shape = (nx, ny)`` instead of ``shape = (ny, nx)``.


    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the grid points
    * ``[x, y, z]``
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points

    Examples::

        >>> x, y = regular((0, 10, 0, 5), (5, 3))
        >>> x
        array([  0. ,   0. ,   0. ,   2.5,   2.5,   2.5,   5. ,   5. ,   5. ,
                 7.5,   7.5,   7.5,  10. ,  10. ,  10. ])
        >>> x.reshape((5, 3))
        array([[  0. ,   0. ,   0. ],
               [  2.5,   2.5,   2.5],
               [  5. ,   5. ,   5. ],
               [  7.5,   7.5,   7.5],
               [ 10. ,  10. ,  10. ]])
        >>> y.reshape((5, 3))
        array([[ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ]])
        >>> x, y = regular((0, 0, 0, 5), (1, 3))
        >>> x.reshape((1, 3))
        array([[ 0.,  0.,  0.]])
        >>> y.reshape((1, 3))
        array([[ 0. ,  2.5,  5. ]])
        >>> x, y, z = regular((0, 10, 0, 5), (5, 3), z=-10)
        >>> z.reshape((5, 3))
        array([[-10., -10., -10.],
               [-10., -10., -10.],
               [-10., -10., -10.],
               [-10., -10., -10.],
               [-10., -10., -10.]])


    """
    nx, ny = shape
    x1, x2, y1, y2 = area
    assert x1 <= x2, \
        "Invalid area dimensions {}, {}. x1 must be < x2.".format(x1, x2)
    assert y1 <= y2, \
        "Invalid area dimensions {}, {}. y1 must be < y2.".format(y1, y2)
    xs = numpy.linspace(x1, x2, nx)
    ys = numpy.linspace(y1, y2, ny)
    # Must pass ys, xs in this order because meshgrid uses the first argument
    # for the columns
    arrays = numpy.meshgrid(ys, xs)[::-1]
    if z is not None:
        arrays.append(z*numpy.ones(nx*ny, dtype=numpy.float))
    return [i.ravel() for i in arrays]



