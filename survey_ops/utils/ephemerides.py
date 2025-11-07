import ephem
from datetime import datetime

def blanco_observer(time=None):
    """
    Define an ephem.Observer for the Blanco telescope location. Location info from:
    https://www.ctio.noirlab.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon

    Arguments
    ---------
    time : float [None]
        Time (Unix timestamp, in UTC) at which to define observer. Default: now.

    Returns
    -------
    observer : ephem.Observer
        Ephem observer object for Blanco location at specified time.
    """

    # define the location
    observer = ephem.Observer()
    observer.lat = "-30:10:10.78" # -30.169661 deg
    observer.lon = "-70:48:23.49" # -70.806525 deg
    observer.elevation = 2206.8  # m

    # initialize time
    if time is None:
        observer.date = datetime.utcnow().strftime("%Y/%m/%d %H:%M:%S")
    else:
        observer.date = datetime.fromtimestamp(time).strftime("%Y/%m/%d %H:%M:%S")

    return observer

def get_source_ra_dec(source, time=None):
    """
    Get the astrophysical coordinates of a known source using pyephem.

    Arguments
    ---------
    src : str
        Source name. Options: "moon", "sun"
    at_time : float [None]
        Time (Unix timestamp, in UTC) at which to determine position. Default: now.

    Returns
    -------
    ra, dec : float
        Source position in radians
    """

    # create observer at Blanco telescope
    observer = blanco_observer(time=time)

    # grab ephem object for the source
    source = source.lower()
    if source == "moon":
        body = ephem.Moon()
    elif source == "sun":
        body = ephem.Sun()
    else:
        raise NotImplementedError("Getting ephemerides for invalid source: " + source)

    # compute source location
    body.compute(observer)
    return body.ra, body.dec

def equatorial_to_topographic(ra, dec, time=None):
    """
    Convert RA/Dec to Az/El for the Blanco telescope location.

    Arguments
    ---------
    ra : float
        Right ascension in radians
    dec : float
        Declination in radians
    time : float [None]
        Time (Unix timestamp, in UTC) at which to determine position. Default: now.

    Returns
    -------
    az, el : float
        Azimuth and elevation in radians
    """

    # initialize observer location and time
    observer = blanco_observer(time=time)

    # define position in equatorial coordinates
    source = ephem.FixedBody()
    source._ra = ra
    source._dec = dec

    # compute topographic position for the observer
    source.compute(observer)
    return source.az, source.alt

def topographic_to_equatorial(az, el, time=None):
    """
    Convert Az/El to RA/Dec for the Blanco telescope location.

    Arguments
    ---------
    az : float
        Azimuth in radians
    el : float
        Elevation in radians
    time : float [None]
        Time (Unix timestamp, in UTC) at which to determine position. Default: now.

    Returns
    -------
    ra, dec : float
        Right ascension and declination in radians
    """

    # initialize observer location and time
    observer = blanco_observer(time=time)

    # compute topographic position for the observer
    return observer.radec_of(az, el)
