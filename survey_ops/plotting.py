from datetime import datetime, timezone
from survey_ops.utils import units, ephemerides
import tempfile, os, shutil, glob
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from tqdm import tqdm


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

    def poly(self, ra, dec, **kwargs):
        """
        Plots a polygon on map.

        Arguments
        ---------
        ra, dec : array of floats
            RA and Dec coordinates to plot
        kwargs
            Options to pass to maplotlib.patches.Polygon
        """
        lon = self.ra_to_lon(ra)
        lat = np.asarray(dec)
        verts = np.column_stack([lon, lat]) / units.deg
        patch = Polygon(verts, closed=True, transform=self.input_crs, **kwargs)
        self.ax.add_patch(patch)

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
    zenith_ra, zenith_dec = ephemerides.get_source_ra_dec("zenith", observer=observer)
    skymap = SkyMap(center_ra=zenith_ra, center_dec=zenith_dec)

    # set title to selected time
<<<<<<< HEAD
    plt.title(datetime.fromtimestamp(time, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S") + " UTC")
=======
    plt.title(
        datetime.fromtimestamp(time, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
        + " UTC"
    )
>>>>>>> f1a31e42cfd35d45b6313beccbafc80d71173a65

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
    for i, (time, ra, dec) in enumerate(zip(tqdm(times), ras, decs)):
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


def plot_bins(
    time,
    current_idx,
    alternate_idx,
    completed_idxs,
    future_idxs,
    nside,
    is_azel=False,
    plot_zenith=True,
    plot_airmass=1.4,
    plot_galaxy=True,
    plot_moon=True,
    sky_bin_mapping=None,
):
    """
    Initialize a sky view binned into healpix grid. Optionally highlight select pixels.

    Arguments
    ---------
    time : float
        Time (Unix timestamp, in UTC) of current observation.
    current_idx : int
        Index of current sky bin, highlighted in green.
    alternate_idx : int
        Index of an alternate sky bin, highlighted in red if different from current. For
        instance, a policy that is being compared to an expert can be compared in red.
    completed_idxs : list of int
        List of sky bin indices with observations yet to be observed.
    future_idxs : list of int
        List of sky bin indices already completed.
    nside : int
        nside used to instantiate the ephemerides.HealpixGrid
    is_azel : bool [False]
        whether the ephemerides.HealpixGrid uses az/el (True) or RA/Dec coords (False)
    plot_zenith : bool [True]
        Whether to plot a circle at zenith
    plot_airmass : float [1.4]
        Plots a circle at constant airmass. Pass 0 for no circle
    plot_galaxy : bool [True]
        Whether to plot a line through the galactic plane +/-1deg
    plot_moon : bool [True]
        Whether to plot a circle at the moon's position
    sky_bin_mapping : dict [None]
        If provided, is used to validate that the recreated healpix grid matches the
        provided grid.

    Returns
    -------
    skymap : SkyMap instance
        The created figure.
    """
    from collections import Counter

    # initialize figure at selected time
    observer = ephemerides.blanco_observer(time=time)
    zenith_ra, zenith_dec = ephemerides.get_source_ra_dec("zenith", observer=observer)
    skymap = SkyMap(center_ra=zenith_ra, center_dec=zenith_dec)

<<<<<<< HEAD
    # set title to selected time #TODO fix timezone issue
    plt.title(datetime.fromtimestamp(time, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S") + " UTC")
=======
    # set title to selected time
    plt.title(
        datetime.fromtimestamp(time, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
        + " UTC"
    )
>>>>>>> f1a31e42cfd35d45b6313beccbafc80d71173a65

    # re-create the healpix grid
    hpgrid = ephemerides.HealpixGrid(nside=nside, is_azel=is_azel)
    if sky_bin_mapping is not None:
        assert len(sky_bin_mapping) == len(hpgrid.lon)
        for idx, (lon, lat) in sky_bin_mapping.items():
            assert int(idx) == hpgrid.ang2idx(lon=lon, lat=lat)

    # plot the healpix grid lines
    ra, dec = hpgrid.get_pixel_boundaries(step=2)
    if is_azel:
        ra, dec = ephemerides.topographic_to_equatorial(
            az=ra, el=dec, observer=observer
        )
    for r, d in zip(ra, dec):
        skymap.plot(ra=r, dec=d, color="pink", linewidth=0.8, linestyle=":")

    # mark the current sky bin
    skymap.poly(
        ra=ra[current_idx],
        dec=dec[current_idx],
        facecolor="none",
        edgecolor="darkgreen",
        zorder=10,
        linewidth=2,
    )

    # mark the alternate sky bin
    if alternate_idx != current_idx:
        skymap.poly(
            ra=ra[alternate_idx],
            dec=dec[alternate_idx],
            facecolor="none",
            edgecolor="red",
            hatch="xxx",
            linewidth=2,
            zorder=10,
        )

    # count up completed and total visits per bin
    completed_counts = Counter(completed_idxs)
    completed_counts[current_idx] += 1
    total_counts = completed_counts + Counter(future_idxs)

    # shade in the sky bins to be visited
    for idx in set(future_idxs) - set(completed_counts):
        skymap.poly(
            ra=ra[idx],
            dec=dec[idx],
            facecolor="silver",
            alpha=0.6,
            zorder=9,
        )

    # plot counts for completed bins
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
        return colors.LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
            cmap(np.linspace(minval, maxval, n)),
        )

    cmap = truncate_colormap(cm.Greens, 0.3, 1.0)
    norm = colors.Normalize(vmin=1, vmax=max(total_counts.values()))
    for idx, count in completed_counts.items():
        if alternate_idx != current_idx and idx == current_idx:
            continue
        skymap.poly(
            ra=ra[idx],
            dec=dec[idx],
            facecolor=cmap(norm(count)),
            alpha=0.6,
            zorder=9,
        )

    # colorbar for completed bins
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = skymap.fig.colorbar(
        sm, ax=skymap.ax, orientation="vertical", fraction=0.046, pad=0.07
    )
    cbar.locator = MaxNLocator(integer=True)
    cbar.set_label("Completed observations")

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


def plot_bins_movie(
    outfile, nside, times, idxs, alternate_idxs=None, sky_bin_mapping=None
):
    """
    Creates a gif of fields observed over the course of a night.

    Arguments
    ---------
    outfile : str
        Path to output gif file.
    nside : int
        nside used to make the ephemerides.HealpixGrid
    times : list of float
        List of times (Unix timestamps, in UTC) of observations.
    idxs : list of float
        List of bin indices for each observation.
    alternate_idxs : list of float [None]
        List of alternate bin indices for each observation. Defaults to copy bins
    sky_bin_mapping : dict [None]
        If provided, is used to validate that the recreated healpix grid matches the
        provided grid.
    """

    # ensure output file is gif
    if os.path.splitext(outfile)[-1] not in [".gif"]:
        raise NotImplementedError("Only animated gif currently supported.")

    # create temporary directory for temporary png files
    tmpdir = tempfile.mkdtemp()

    # duplicate bins if alternate bins is not given
    alternate_idxs = alternate_idxs if alternate_idxs is not None else idxs

    # plot each observation successively, saving pngs
    plt.ioff()
<<<<<<< HEAD
    for i, (time, idx, alternate_idx) in tqdm(enumerate(zip(times, idxs, alternate_idxs)), total=len(times)):
=======
    for i, (time, idx, alternate_idx) in enumerate(
        zip(tqdm(times), idxs, alternate_idxs)
    ):
>>>>>>> f1a31e42cfd35d45b6313beccbafc80d71173a65
        skymap = plot_bins(
            time,
            current_idx=idx,
            alternate_idx=alternate_idx,
            completed_idxs=idxs[:i],
            future_idxs=idxs[i + 1 :],
            nside=nside,
            sky_bin_mapping=sky_bin_mapping,
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
    parser.add_argument(
        "-b",
        "--bin_schedule_file",
        type=str,
        help='Path to the sky bin index schedule file, a csv with keys "time", "policy_bin_id", and "bin_id".',
    )
    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        help="nside used to make healpix sky bin for bin schedules",
    )
    args = parser.parse_args()

    # load field file
    with open(args.field_file) as f:
        id2pos = json.load(f)

    # call plotting functions
    if args.schedule_file:
        schedule = pd.read_csv(args.schedule_file)
        plot_fields_movie(
            outfile=args.outfile,
            times=schedule["time"].values,
            ras=[
                id2pos[str(fid)][0] * units.deg for fid in schedule["field_id"].values
            ],
            decs=[
                id2pos[str(fid)][1] * units.deg for fid in schedule["field_id"].values
            ],
        )
    elif args.bin_schedule_file:
        schedule = pd.read_csv(args.bin_schedule_file)
        plot_bins_movie(
            outfile=args.outfile,
            nside=args.nside,
            times=schedule["time"].values,
            idxs=schedule["bin_id"].values,
            alternate_idxs=schedule["policy_bin_id"].values,
            sky_bin_mapping=id2pos,
        )
