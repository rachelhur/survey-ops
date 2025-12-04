from datetime import datetime
from survey_ops.utils import units, ephemerides
import tempfile, os, shutil, glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class SkyMap:
    """
    A class for making sky maps in RA and Dec using Cartopy. Cartopy expects lon and lat
    coordinates, while RA increases opposite to lon. This class supports orthographic,
    Mollweide, Hammer, and Aitoff projections, and it can plot scatters and lines.
    """
    
    def __init__(
        self, center_ra=0, center_dec=0, projection="ortho", figsize=(10.5,8.5), dpi=80
    ):
        """
        Initialize the class.

        Arguments:
        center_ra, center_dec : float
            Center RA, Dec coordinates in radians.
        projection : str
            Map projection to use
        figsize : tuple of floats
            figsize to pass to matplotlib
        dpi : int
            dpi to pass to matplotlib
        """

        # convert RA center to cartopy longitude
        self.center_lon = - center_ra / units.deg % 360
        self.center_lat = center_dec / units.deg

        # choose a projection
        if projection.lower() in ["ortho", "orthographic"]:
            self.projection = ccrs.Orthographic(
                central_longitude=self.center_lon,
                central_latitude=self.center_lat
            )
        elif projection.lower() in ["moll", "mollweide"]:
            self.projection = ccrs.Mollweide(central_longitude=self.center_lon)
        elif projection.lower() in ["hammer"]:
            self.projection = ccrs.Hammer(central_longitude=self.center_lon)
        elif projection.lower() in ["aitoff"]:
            self.projection = ccrs.Aitoff(central_longitude=self.center_lon)
        else:
            raise ValueError(f"Unknown projection: {projection}")

        # set up figure and axes
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=self.projection)
        self.ax.set_global()

        # CRS for RA/Dec inputs
        self.input_crs = ccrs.PlateCarree()

        # initialize plot gridlines
        self.ax.gridlines(color='gray', linestyle='dotted', linewidth=0.8)


    # convert RA (in rad) to the longitude convention (in rad) Cartopy expects
    @staticmethod
    def ra_to_lon(ra_deg):
        return (-np.asarray(ra_deg)) % (360 * units.deg)


    # public plotting API ---------------------------------

    def scatter(self, ra, dec, **kwargs):
        """
        Plots points on map as a scatter plot.

        Arguments:
        ra, dec : array of floats
            RA and Dec coordinates to plot
        kwargs :
            Options to pass to ax.scatter
        """
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        self.ax.scatter(
            lon / units.deg, lat / units.deg, transform=self.input_crs, **kwargs
        )


    def plot(self, ra, dec, **kwargs):
        """
        Plots points on map as a line plot.

        Arguments:
        ra, dec : array of floats
            RA and Dec coordinates to plot
        kwargs :
            Options to pass to ax.plot
        """
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        self.ax.plot(
            lon / units.deg, lat / units.deg, transform=self.input_crs, **kwargs
        )


    def text(self, ra, dec, label, **kwargs):
        """
        Adds text to the map.

        Arguments:
        ra, dec : array of floats
            RA and Dec coordinates to add text
        label : str
            test to add
        kwargs :
            Options to pass to ax.text
        """
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        self.ax.text(
            lon / units.deg, lat / units.deg, label, transform=self.input_crs, **kwargs
        )


def plot_fields(time, current_radec, completed_radec, future_radec):
    """
    Initialize a sky view and plot of current and completed fields at a selected time.

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
    skymap : SkyMap instance
        The created figure.
    """

    # initialize figure at selected time
    observer = ephemerides.blanco_observer(time=time)
    zenith_ra, zenith_dec = ephemerides.topographic_to_equatorial(
        0, '90', observer=observer
    )
    skymap = SkyMap(center_ra=zenith_ra, center_dec=zenith_dec)

    # set title to selected time
    plt.title(datetime.fromtimestamp(time).strftime("%Y/%m/%d %H:%M:%S") + " UTC")

    # plot current field
    skymap.scatter(
        ra=current_radec[0],
        dec=current_radec[1],
        c='0.6',
        edgecolor='k',
        zorder=10,
        marker='H',
        s=80,
    )

    # plot completed fields
    if len(completed_radec) > 0:
        skymap.scatter(
            ra=np.asarray(completed_radec)[:,0],
            dec=np.asarray(completed_radec)[:,1],
            c='0.8',
            edgecolor=None,
            zorder=9,
            marker='H',
            s=80,
        )

    # plot future fields
    if len(future_radec) > 1:
        skymap.scatter(
            ra=np.asarray(future_radec)[:,0],
            dec=np.asarray(future_radec)[:,1],
            c='1.0',
            edgecolor='gainsboro',
            zorder=8,
            marker='H',
            s=80,
        )

    return skymap

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
        skymap = plot_fields(
            time,
            current_radec=(ra, dec),
            completed_radec=list(zip(ras[:i], decs[:i])),
            future_radec=list(zip(ras[i+1:], decs[i+1:]))
        )
        plt.savefig(os.path.join(tmpdir, 'field_%08i.png' % i))
        plt.close(skymap.fig)
    plt.ion()

    # convert pngs to gif
    pngs = sorted(glob.glob(os.path.join(tmpdir, '*.png')))
    if not pngs:
        shutil.rmtree(tmpdir)
        raise RuntimeError("No PNG frames were generated for plot_fields_movie()")
    images = [imageio.imread(p) for p in pngs]
    imageio.mimsave(outfile, images, duration=0.10, loop=0)
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
