from datetime import datetime
from survey_ops.utils import units, ephemerides
import tempfile, os, shutil, glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patheffects as pe


class SkyMap:
    """
    A class for making sky maps in RA and Dec using Cartopy. Cartopy expects lon and lat
    coordinates, while RA increases opposite to lon. This class supports orthographic,
    Mollweide, Hammer, and Aitoff projections, and it can plot scatters and lines.
    """

    def __init__(
        self, center_ra=0, center_dec=0, projection="ortho", figsize=(10.5, 8.5), dpi=80
    ):
        """
        Initialize the class.

        Arguments
        ---------
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
        self.center_lon = -center_ra / units.deg % 360
        self.center_lat = center_dec / units.deg

        # choose a projection
        if projection.lower() in ["ortho", "orthographic"]:
            self.projection = ccrs.Orthographic(
                central_longitude=self.center_lon, central_latitude=self.center_lat
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
        self.ax.gridlines(color="gray", linestyle="dotted", linewidth=0.8)

        # add labels to ra gridlines
        for ra in np.arange(0, 360, 60) * units.deg:
            self.text(
                ra=ra,
                dec=-7 * units.deg,
                label=f"{ra / units.deg:.0f}°",
                outline="white",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                rotation=90,
            )

        # add labels to dec gridlines
        for dec in np.arange(-80, 80 + 1e-5, 20) * units.deg:
            for ra in np.array([10, 190]) * units.deg:
                self.text(
                    ra=ra,
                    dec=dec,
                    label=f"{dec / units.deg:.0f}°",
                    outline="white",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    rotation=0,
                )

        # add East and West labels
        self.ax.annotate(
            "West", xy=(1.01, 0.5), ha="left", xycoords="axes fraction", rotation=-90
        )
        self.ax.annotate(
            "East", xy=(-0.01, 0.5), ha="right", xycoords="axes fraction", rotation=90
        )

    # convert RA (in rad) to the longitude convention (in rad) Cartopy expects
    @staticmethod
    def ra_to_lon(ra_deg):
        return np.unwrap(-np.atleast_1d(ra_deg))

    # public plotting API ---------------------------------

    def scatter(self, ra, dec, **kwargs):
        """
        Plots points on map as a scatter plot.

        Arguments
        ---------
        ra, dec : array of floats
            RA and Dec coordinates to plot
        kwargs
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

        Arguments
        ---------
        ra, dec : array of floats
            RA and Dec coordinates to plot
        kwargs
            Options to pass to ax.plot
        """
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        self.ax.plot(
            lon / units.deg, lat / units.deg, transform=self.input_crs, **kwargs
        )

    def text(self, ra, dec, label, outline=None, **kwargs):
        """
        Adds text to the map.

        Arguments
        ---------
        ra, dec : array of floats
            RA and Dec coordinates to add text
        label : str
            test to add
        outline : str
            If provided, the color to outline the text in
        kwargs
            Options to pass to ax.text
        """
        # draw the text
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        txt = self.ax.text(
            lon / units.deg, lat / units.deg, label, transform=self.input_crs, **kwargs
        )

        # outline the text
        if outline is not None:
            txt.set_path_effects([pe.withStroke(linewidth=5, foreground=outline)])


def plot_fields(
    time,
    current_radec,
    completed_radec,
    future_radec,
    plot_zenith=True,
    plot_airmass=1.4,
    plot_galaxy=True,
    plot_moon=True,
):
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
    plot_zenith : bool [True]
        Whether to plot a circle at zenith
    plot_airmass : float [1.4]
        Plots a circle at constant airmass. Pass 0 for no circle
    plot_galaxy : bool [True]
        Whether to plot a line through the galactic plane +/-1deg
    plot_moon : bool [True]
        Whether to plot a circle at the moon's position

    Returns
    -------
    skymap : SkyMap instance
        The created figure.
    """

    # initialize figure at selected time
    observer = ephemerides.blanco_observer(time=time)
    zenith_ra, zenith_dec = ephemerides.topographic_to_equatorial(
        0, "90", observer=observer
    )
    skymap = SkyMap(center_ra=zenith_ra, center_dec=zenith_dec)

    # set title to selected time
    plt.title(datetime.fromtimestamp(time).strftime("%Y/%m/%d %H:%M:%S") + " UTC")

    # plot current field
    skymap.scatter(
        ra=current_radec[0],
        dec=current_radec[1],
        c="0.6",
        edgecolor="k",
        zorder=10,
        marker="H",
        s=80,
    )

    # plot completed fields
    if len(completed_radec) > 0:
        skymap.scatter(
            ra=np.asarray(completed_radec)[:, 0],
            dec=np.asarray(completed_radec)[:, 1],
            c="0.8",
            edgecolor=None,
            zorder=9,
            marker="H",
            s=80,
        )

    # plot future fields
    if len(future_radec) > 1:
        skymap.scatter(
            ra=np.asarray(future_radec)[:, 0],
            dec=np.asarray(future_radec)[:, 1],
            facecolor="none",
            edgecolor="gainsboro",
            zorder=8,
            marker="H",
            s=80,
        )

    # plot zenith marking
    if plot_zenith:
        skymap.scatter(
            ra=zenith_ra,
            dec=zenith_dec,
            facecolor="none",
            edgecolor="green",
            zorder=10,
            marker="o",
            s=80,
        )

    # plot lines through the galactic plane
    if plot_galaxy:
        l = np.linspace(0, 360, 100) * units.deg
        b = np.zeros_like(l)
        ra, dec = ephemerides.galactic_to_equatorial(l=l, b=b)
        skymap.plot(ra=ra, dec=dec, color="gray", zorder=10, linewidth=0.8)
        for offset in [5 * units.deg, -5 * units.deg]:
            ra, dec = ephemerides.galactic_to_equatorial(l=l, b=b + offset)
            skymap.plot(
                ra=ra, dec=dec, color="gray", zorder=10, linestyle="--", linewidth=0.8
            )

    # plot requested airmass
    if plot_airmass != 0:
        az = np.linspace(0, 360, 100) * units.deg
        el = np.ones_like(az) * 90 * units.deg - np.arccos(1 / plot_airmass)
        ra, dec = ephemerides.topographic_to_equatorial(az=az, el=el, observer=observer)
        skymap.plot(
            ra=ra,
            dec=dec,
            color="green",
            zorder=10,
            linewidth=0.8,
        )

    # plot the moon
    if plot_moon:
        ra, dec = ephemerides.get_source_ra_dec(source="moon", observer=observer)
        skymap.scatter(
            ra=ra,
            dec=dec,
            facecolor="darkgrey",
            edgecolor="black",
            marker="o",
            s=300,
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
    if os.path.splitext(outfile)[-1] not in [".gif"]:
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
            future_radec=list(zip(ras[i + 1 :], decs[i + 1 :])),
        )
        plt.savefig(os.path.join(tmpdir, "field_%08i.png" % i))
        plt.close(skymap.fig)
    plt.ion()

    # convert pngs to gif
    pngs = sorted(glob.glob(os.path.join(tmpdir, "*.png")))
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
        help='Path to the schedule file, a csv with keys "time" and "field_id".',
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
        times=schedule["time"].values,
        ras=[id2pos[str(fid)][0] * units.deg for fid in schedule["field_id"].values],
        decs=[id2pos[str(fid)][1] * units.deg for fid in schedule["field_id"].values],
    )
