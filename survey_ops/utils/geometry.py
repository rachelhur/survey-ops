import numpy as np


def angular_separation(pos1, pos2):
    """
    Calculate the angular distance between two points on a sphere using the special case
    of the Vincenty formula for a sphere.

    Arguments
    ---------
    pos1, pos2 : float tuple
        The (horizontal, vertical) positions of the two points on the sphere, provided
        in radians. For example: (ra, dec), (lat, lon), (az, el), etc.

    Returns
    -------
    distance : float
        The angular distance in radians between the two specified points.
    """

    # extract az and el from tuples
    az1, el1 = pos1
    az2, el2 = pos2

    # precalculate repeated terms
    daz = np.abs(az2 - az1)
    cos_daz = np.cos(daz)
    cos_el1 = np.cos(el1)
    sin_el1 = np.sin(el1)
    cos_el2 = np.cos(el2)
    sin_el2 = np.sin(el2)

    # use Vincenty formula
    return np.arctan2(
        np.sqrt(
            (cos_el2 * np.sin(daz)) ** 2
            + (cos_el1 * sin_el2 - sin_el1 * cos_el2 * cos_daz) ** 2
        ),
        sin_el1 * sin_el2 + cos_el1 * cos_el2 * cos_daz,
    )
