"""
This code presents a general approach for implementing the gravitational
potential and vertical component of the gravitational acceleration produced
by a rectangular prism by using the analytical formulas of
Nagy et al (2000, 2002). This prototype is inspired on
[Harmonica](https://www.fatiando.org/harmonica/latest/index.html)
(Uieda et al, 2020). It makes use of the modified arctangent function proposed
by Fukushima (2020, eq. 72) and of a modified logarithm function for dealing
with singularities at some computation points.
"""


import numpy as np
from numba import njit
from .. import check
from .. import utils
from .. import constants as cts


def grav(coordinates, prisms, density, field):
    """
    Gravitational potential, first and second derivatives
    produced by a right-rectangular prism in Cartesian coordinates.
    All values are referred to a topocentric Cartesian system with axes
    x, y and z pointing to north, east and down, respectively.

    Parameters
    ----------
    coordinates : 2d-array
        2d-array containing x (first line), y (second line), and z (third line) of
        the computation points. All coordinates should be in meters.
    prisms : 2d-array
        2d-array containing the coordinates of the prisms. Each line must contain
        the coordinates of a single prism in the following order:
        south (x1), north (x2), west (y1), east (y2), top (z1) and bottom (z2).
        All coordinates should be in meters.
    density : 1d-array
        1d-array containing the density of each prism in kg/m^3.
    field : str
        Gravitational field to be computed.
        The available fields are:

        - Gravitational potential: ``g_potential`` (in m² / s²)
        - z-component of acceleration: ``g_z`` (in mGal)
        - y-component of acceleration: ``g_y`` (in mGal)
        - x-component of acceleration: ``g_x`` (in mGal)
        - zz-component of acceleration: ``g_zz`` (in Eötvös)
        - yz-component of acceleration: ``g_yz`` (in Eötvös)
        - xz-component of acceleration: ``g_xz`` (in Eötvös)
        - yy-component of acceleration: ``g_yy`` (in Eötvös)
        - xy-component of acceleration: ``g_xy`` (in Eötvös)
        - xx-component of acceleration: ``g_xx`` (in Eötvös)

    Returns
    -------
    result : array
        Gravitational field generated by the prisms at the computation points.

    """

    # Available fields
    fields = {
        "g_potential": kernel_inverse_r,
        "g_x": kernel_dx,
        "g_y": kernel_dy,
        "g_z": kernel_dz,
        "g_xx": kernel_dxx,
        "g_xy": kernel_dxy,
        "g_xz": kernel_dxz,
        "g_yy": kernel_dyy,
        "g_yz": kernel_dyz,
        "g_zz": kernel_dzz,
    }

    # Verify the field
    if field not in fields:
        raise ValueError("Gravitational field {} not recognized".format(field))

    # Verify the input parameters
    check.coordinates(coordinates)
    check.rectangular_prisms(prisms)
    check.density(density, prisms)

    # create the array to store the result
    result = np.zeros(coordinates[0].size, dtype="float64")

    # Compute gravitational field
    jit_grav(coordinates, prisms, density, fields[field], result)
    result *= cts.GRAVITATIONAL_CONST
    # Convert from m/s^2 to mGal
    if field in ["g_x", "g_y", "g_z"]:
        result *= cts.SI2MGAL
    # Convert from 1/s^2 to Eötvös
    if field in ["g_xx", "g_xy", "g_xz", "g_yy", "g_yz", "g_zz"]:
        result *= cts.SI2EOTVOS
    return result


def mag(coordinates, prisms, magnetization, field):
    """
    Magnetic scalar potential and magnetic induction components
    produced by a right-rectangular prism in Cartesian coordinates.
    All values are referred to a topocentric Cartesian system with axes
    x, y and z pointing to north, east and down, respectively.

    Parameters
    ----------
    coordinates : 2d-array
        2d-array containing x (first line), y (second line), and z (third line) of
        the computation points. All coordinates should be in meters.
    prisms : 2d-array
        2d-array containing the coordinates of the prisms. Each line must contain
        the coordinates of a single prism in the following order:
        south (x1), north (x2), west (y1), east (y2), top (z1) and bottom (z2).
        All coordinates should be in meters.
    magnetization : 2d-array
        2d-array containing the total-magnetization components of the prisms.
        Each line must contain the intensity (in A/m), inclination and
        declination (in degrees) of the total magnetization of a single prism.
    field : str
        Magnetic field to be computed.
        The available fields are:

        - Magnetic scalar potential: ``b_potential`` (in uT x m)
        - z-component of induction: ``b_z`` (in nT)
        - y-component of induction: ``b_y`` (in nT)
        - x-component of induction: ``b_x`` (in nT)

    Returns
    -------
    result : array
        Magnetic field generated by the prisms at the computation points.

    """

    # Available fields
    fields = {
        "b_potential": {"x": kernel_dx, "y": kernel_dy, "z": kernel_dz},
        "b_z": {"x": kernel_dxz, "y": kernel_dyz, "z": kernel_dzz},
        "b_y": {"x": kernel_dxy, "y": kernel_dyy, "z": kernel_dyz},
        "b_x": {"x": kernel_dxx, "y": kernel_dxy, "z": kernel_dxz},
    }

    # Verify the field
    if field not in fields:
        raise ValueError("Magnetic field {} not recognized".format(field))

    # Verify the input parameters
    check.coordinates(coordinates)
    check.rectangular_prisms(prisms)
    check.magnetization(magnetization, prisms)

    # create the array to store the result
    result = np.zeros(coordinates[0].size, dtype="float64")

    # Compute the Cartesian components of total-magnetization
    mx, my, mz = utils.magnetization_components(magnetization)

    # Compute magnetic field
    fieldx = fields[field]["x"]
    fieldy = fields[field]["y"]
    fieldz = fields[field]["z"]
    jit_mag(coordinates, prisms, mx, my, mz, fieldx, fieldy, fieldz, result)
    result *= cts.CM
    # Convert from T to nT
    if field in ["b_x", "b_y", "b_z"]:
        result *= cts.T2NT
    # Convert from T to uT and change sign
    if field == "b_potential":
        result *= -cts.T2MT
    return result


@njit
def jit_grav(coordinates, prisms, density, field, out):
    """
    Compute the gravitational field at the points in 'coordinates'
    """
    # Iterate over computation points
    for l in range(coordinates[0].size):
        # Iterate over prisms
        for p in range(prisms.shape[0]):
            # Change coordinates
            X1 = prisms[p, 0] - coordinates[0, l]
            X2 = prisms[p, 1] - coordinates[0, l]
            Y1 = prisms[p, 2] - coordinates[1, l]
            Y2 = prisms[p, 3] - coordinates[1, l]
            Z1 = prisms[p, 4] - coordinates[2, l]
            Z2 = prisms[p, 5] - coordinates[2, l]
            # Compute the field
            out[l] += density[p] * (
                field(X2, Y2, Z2)
                - field(X2, Y2, Z1)
                - field(X1, Y2, Z2)
                + field(X1, Y2, Z1)
                - field(X2, Y1, Z2)
                + field(X2, Y1, Z1)
                + field(X1, Y1, Z2)
                - field(X1, Y1, Z1)
            )


@njit
def jit_mag(coordinates, prisms, mx, my, mz, fieldx, fieldy, fieldz, out):
    """
    Compute the magnetic field at the points in 'coordinates'
    """
    # Iterate over computation points
    for l in range(coordinates[0].size):
        # Iterate over prisms
        for p in range(prisms.shape[0]):
            # Change coordinates
            X1 = prisms[p, 0] - coordinates[0, l]
            X2 = prisms[p, 1] - coordinates[0, l]
            Y1 = prisms[p, 2] - coordinates[1, l]
            Y2 = prisms[p, 3] - coordinates[1, l]
            Z1 = prisms[p, 4] - coordinates[2, l]
            Z2 = prisms[p, 5] - coordinates[2, l]
            # Compute the field component x
            out[l] += mx[p] * (
                fieldx(X2, Y2, Z2)
                - fieldx(X2, Y2, Z1)
                - fieldx(X1, Y2, Z2)
                + fieldx(X1, Y2, Z1)
                - fieldx(X2, Y1, Z2)
                + fieldx(X2, Y1, Z1)
                + fieldx(X1, Y1, Z2)
                - fieldx(X1, Y1, Z1)
            )
            # Compute the field component y
            out[l] += my[p] * (
                fieldy(X2, Y2, Z2)
                - fieldy(X2, Y2, Z1)
                - fieldy(X1, Y2, Z2)
                + fieldy(X1, Y2, Z1)
                - fieldy(X2, Y1, Z2)
                + fieldy(X2, Y1, Z1)
                + fieldy(X1, Y1, Z2)
                - fieldy(X1, Y1, Z1)
            )
            # Compute the field component z
            out[l] += mz[p] * (
                fieldz(X2, Y2, Z2)
                - fieldz(X2, Y2, Z1)
                - fieldz(X1, Y2, Z2)
                + fieldz(X1, Y2, Z1)
                - fieldz(X2, Y1, Z2)
                + fieldz(X2, Y1, Z1)
                + fieldz(X1, Y1, Z2)
                - fieldz(X1, Y1, Z1)
            )


# kernels


@njit
def kernel_inverse_r(X, Y, Z):
    """
    Function for computing the inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = (
        Y * X * utils.safe_log(Z + R)
        + X * Z * utils.safe_log(Y + R)
        + Y * Z * utils.safe_log(X + R)
        - 0.5 * Y ** 2 * utils.safe_atan2(Z * X, Y * R)
        - 0.5 * X ** 2 * utils.safe_atan2(Z * Y, X * R)
        - 0.5 * Z ** 2 * utils.safe_atan2(Y * X, Z * R)
    )
    return result


@njit
def kernel_dz(X, Y, Z):
    """
    Function for computing the z-derivative of inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = -(
        Y * utils.safe_log(X + R)
        + X * utils.safe_log(Y + R)
        - Z * utils.safe_atan2(Y * X, Z * R)
    )
    return result


@njit
def kernel_dy(X, Y, Z):
    """
    Function for computing the y-derivative of inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = -(
        X * utils.safe_log(Z + R)
        + Z * utils.safe_log(X + R)
        - Y * utils.safe_atan2(X * Z, Y * R)
    )
    return result


@njit
def kernel_dx(X, Y, Z):
    """
    Function for computing the x-derivative of inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = -(
        Y * utils.safe_log(Z + R)
        + Z * utils.safe_log(Y + R)
        - X * utils.safe_atan2(Y * Z, X * R)
    )
    return result


@njit
def kernel_dzz(X, Y, Z):
    """
    Function for computing the zz-derivative of inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = -utils.safe_atan2(Y * X, Z * R)
    return result


@njit
def kernel_dyz(X, Y, Z):
    """
    Function for computing the yz-derivative of inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = utils.safe_log(X + R)
    return result


@njit
def kernel_dxz(X, Y, Z):
    """
    Function for computing the xz-derivative of inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = utils.safe_log(Y + R)
    return result


@njit
def kernel_dyy(X, Y, Z):
    """
    Function for computing the yy-derivative of inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = -utils.safe_atan2(X * Z, Y * R)
    return result


@njit
def kernel_dxy(X, Y, Z):
    """
    Function for computing the xy-derivative of inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = utils.safe_log(Z + R)
    return result


@njit
def kernel_dxx(X, Y, Z):
    """
    Function for computing the xx-derivative of inverse distance kernel
    """
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    result = -utils.safe_atan2(Y * Z, X * R)
    return result
