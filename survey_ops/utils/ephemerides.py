import ephem
from datetime import datetime

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

    # define BLANCO observer location and time
    # http://www.ctio.noao.edu/noao/content/coordinates-observatories-cerro-tololo-and-cerro-pachon
    observer = ephem.Observer()
    observer.lat = '-30:10:10.78'
    observer.lon = '-70:48:23.49'
    observer.elevation = 2206.8 # m
    if time is None:
        observer.date = datetime.utcnow().strftime("%Y/%m/%d %H:%M:%S")
    else:
        observer.date = datetime.fromtimestamp(time).strftime("%Y/%m/%d %H:%M:%S")

    # grab ephem object for the source
    body = None
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
