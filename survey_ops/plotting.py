from datetime import datetime
from survey_ops.utils import units, ephemerides
import tempfile, os, shutil, glob, imageio
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def initialize_orthographic_plot(time, figsize=(10.5,8.5), dpi=80):
    """
    Initialize a map in orthographic projection for plotting DECam surveys.
    Code adapted from kadrlica's obztak/utils/ortho.py

    Arguments
    ---------
    time : float [None]
        Time (Unix timestamp, in UTC) at which to define observer.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    basemap : mpl_toolkits.basemap.Basemap
        The created basemap object.
    observer : ephem.Observer
        The observer object at the specified time.
    """

    # create an observer at selected time
    observer = ephemerides.blanco_observer(time=time)

    # projection info
    zenith_lon, zenith_lat = ephemerides.topographic_to_equatorial(
        0, '90', observer=observer
    )
    defaults = dict(
        projection='ortho',
        celestial=True,
        rsphere=1.0,
        lon_0= -zenith_lon / units.deg,
        lat_0= zenith_lat / units.deg,
    )

    # initialize the figure with orthographic projection
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.cla()
    basemap = Basemap(**defaults)

    # set title to chosen time
    plt.title(datetime.fromtimestamp(time).strftime("%Y/%m/%d %H:%M:%S") + " UTC")

    return fig, basemap, observer

def plot_field(ra, dec, basemap, observer, status):
    """
    Plot a specific field on an existing orthographic basemap.

    Arguments
    ---------
    ra, dec : float
        Right ascension and declination in radians of the field to plot.
    basemap : mpl_toolkits.basemap.Basemap
        The basemap object.
    observer : ephem.Observer
        The observer object.
    status : str
        Whether the field is "current", "completed", or "future".

    Returns
    -------
    basemap : mpl_toolkits.basemap.Basemap
        The updated basemap object.
    """

    # determine if field is completed
    if status not in ["current", "completed", "future"]:
        raise ValueError("Invalid field status: " + status)
    if status == "completed":
        c = '0.8'
        edgecolor = None
        zorder = 9
    elif status == "current":
        c = '0.6'
        edgecolor = "k"
        zorder = 10
    else:  # future
        c = '1.0'
        edgecolor = "gainsboro"
        zorder = 8

    # pixel location of the field
    x, y = basemap(ra / units.deg, dec / units.deg)

    # plot the field
    basemap.scatter(
        x=x,
        y=y,
        c=c,
        edgecolor=edgecolor,
        marker='H',
        s=80,
        zorder=zorder,
    )

    return basemap

def plot_fields(time, current_radec, completed_radec, future_radec):
    """
    Creates a plot of current and completed fields.

    Arguments
    ---------
    time : float
        Time (Unix timestamp, in UTC) of current observation.
    current_radec : float tuple
        (ra, dec) tuple in radians for current field to plot.
    completed_radec : list of float tuples
        List of (ra, dec) tuples in radians for completed fields to plot.
    future_radec : list of float tuples
        List of (ra, dec) tuples in radians for future fields to plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    basemap : mpl_toolkits.basemap.Basemap
        The basemap object.
    """
    # initialize figure
    fig, basemap, observer = initialize_orthographic_plot(time=time)

    # plot all fields
    basemap = plot_field(
        ra=current_radec[0],
        dec=current_radec[1],
        basemap=basemap,
        observer=observer,
        status="current",
    )
    for ra, dec in completed_radec:
        basemap = plot_field(
            ra=ra,
            dec=dec,
            basemap=basemap,
            observer=observer,
            status="completed",
        )
    for ra, dec in future_radec:
        basemap = plot_field(
            ra=ra,
            dec=dec,
            basemap=basemap,
            observer=observer,
            status="future",
        )

    return fig, basemap

def plot_fields_movie(outfile, times, ras, decs):
    """
    Creates a gif of fields observed over the course of a night.

    Arguments
    ---------
    outfile : str
        Path to output gif file.
    times : list of float
        List of times (Unix timestamps, in UTC) of observations.
    ras : list of float
        List of right ascensions (in radians) for each observation.
    decs : list of float
        List of declinations (in radians) for each observation.
    """

    # ensure output file is gif
    if os.path.splitext(outfile)[-1] not in ['.gif']:
        raise NotImplementedError("Only animated gif currently supported.")

    # create temporary directory for temporary png files
    tmpdir = tempfile.mkdtemp()

    # plot each observation successively, saving pngs
    plt.ioff()
    for i, (time, ra, dec) in enumerate(zip(times, ras, decs)):
        fig, basemap = plot_fields(
            time,
            current_radec=(ra, dec),
            completed_radec=list(zip(ras[:i], decs[:i])),
            future_radec=list(zip(ras[i+1:], decs[i+1:]))
        )
        plt.savefig(os.path.join(tmpdir, 'field_%08i.png' % i))
        plt.close(fig)
    plt.ion()

    # convert pngs to gif
    pngs = sorted(glob.glob(os.path.join(tmpdir, '*.png')))
    if not pngs:
        shutil.rmtree(tmpdir)
        raise RuntimeError("No PNG frames were generated for plot_fields_movie()")
    images = [imageio.imread(p) for p in pngs]
    imageio.mimsave(outfile, images, duration=0.10)
    shutil.rmtree(tmpdir)

    return

if __name__ == "__main__":
    import json
    import argparse as ap
    import pandas as pd

    # command line arguments
    parser = ap.ArgumentParser(
        description="Create a gif of observed fields over the course of a night."
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to output gif file.",
    )
    parser.add_argument(
        "-f",
        "--field_file",
        type=str,
        help="Path to the field file, a json file that maps field_id to (ra, dec).",
    )
    parser.add_argument(
        "-s",
        "--schedule_file",
        type=str,
        help='Path to the schedule file, a csv with keys "time" and "field_id".'
    )
    args = parser.parse_args()

    # load field file
    with open(args.field_file) as f:
        id2pos = json.load(f)

    # load schedule file
    schedule = pd.read_csv(args.schedule_file)

    # call plotting function
    plot_fields_movie(
        outfile=args.outfile,
        times=schedule['time'].values,
        ras=[id2pos[str(fid)][0] * units.deg for fid in schedule['field_id'].values],
        decs=[id2pos[str(fid)][1] * units.deg for fid in schedule['field_id'].values],
    )
