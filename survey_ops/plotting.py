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
    observer=None,
    skymap=None,
    current_kwargs={},
    completed_kwargs={},
    future_kwargs={},
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
    observer : ephem.Observer [None]
        An observer instance. Defaults to blanco_observer at specified time.
    skymap : SkyMap [None]
        A SkyMap plot instance on which to plot. Defaults to creating a new instance.
    current_kwargs, completed_kwargs, future_kwargs : dict [{}]
        kwargs to pass to SkyMap.scatter while plotting fields, updating the defaults.

    Returns
    -------
    skymap : SkyMap instance
        The created figure.
    """

    # initialize figure at selected time
    observer = ephemerides.blanco_observer(time=time) if observer is None else observer
    zenith_ra, zenith_dec = ephemerides.get_source_ra_dec("zenith", observer=observer)
    skymap = (
        SkyMap(center_ra=zenith_ra, center_dec=zenith_dec) if skymap is None else skymap
    )

    # set title to selected time
    plt.title(
        datetime.fromtimestamp(time, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
        + " UTC"
    )

    # plot current field
    if not np.any(np.isnan(np.asarray(current_radec, dtype=float))):
        kwargs = dict(
            c="0.6",
            edgecolor="k",
            zorder=10,
            marker="H",
            s=80,
        )
        kwargs.update(current_kwargs)
        skymap.scatter(ra=current_radec[0], dec=current_radec[1], **kwargs)

    # plot completed fields
    kwargs = dict(
        c="0.8",
        edgecolor=None,
        zorder=9,
        marker="H",
        s=80,
    )
    kwargs.update(completed_kwargs)
    completed_radec = np.asarray(completed_radec, dtype=float)
    keep = np.all(~np.isnan(completed_radec), axis=1)
    if len(completed_radec) > 0:
        skymap.scatter(
            ra=completed_radec[keep, 0], dec=completed_radec[keep, 1], **kwargs
        )

    # plot future fields
    kwargs = dict(
        facecolor="none",
        edgecolor="gainsboro",
        zorder=8,
        marker="H",
        s=80,
    )
    kwargs.update(future_kwargs)
    future_radec = np.asarray(future_radec, dtype=float)
    keep = np.all(~np.isnan(future_radec), axis=1)
    if len(future_radec) > 1:
        skymap.scatter(ra=future_radec[:, 0], dec=future_radec[:, 1], **kwargs)

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


def plot_fields_movie(outfile, times, field_pos):
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

    # ensure positions are numpy array
    field_pos = np.asarray(field_pos)

    # plot each observation successively, saving pngs
    plt.ioff()
    for i, time in enumerate(tqdm(times)):
        skymap = plot_fields(
            time,
            current_radec=field_pos[i, :],
            completed_radec=field_pos[:i, :],
            future_radec=field_pos[i + 1 :, :],
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

    # set title to selected time
    plt.title(
        datetime.fromtimestamp(time, tz=timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
        + " UTC"
    )

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
    outfile,
    nside,
    times,
    idxs,
    alternate_idxs=None,
    sky_bin_mapping=None,
    field_pos=None,
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
    field_pos : list of float tuples [None]
        List of field (ra, dec) for each observation. If provided, plots specific fields
        overlaid on the bins
    """

    # ensure output file is gif
    if os.path.splitext(outfile)[-1] not in [".gif"]:
        raise NotImplementedError("Only animated gif currently supported.")

    # create temporary directory for temporary png files
    tmpdir = tempfile.mkdtemp()

    # duplicate bins if alternate bins is not given
    alternate_idxs = alternate_idxs if alternate_idxs is not None else idxs

    # make sure field pos is numpy array
    field_pos = None if field_pos is None else np.asarray(field_pos)

    # plot each observation successively, saving pngs
    plt.ioff()
    for i, (time, idx, alternate_idx) in enumerate(
        zip(tqdm(times), idxs, alternate_idxs)
    ):
        # plot the sky bins
        skymap = plot_bins(
            time,
            current_idx=idx,
            alternate_idx=alternate_idx,
            completed_idxs=idxs[:i],
            future_idxs=idxs[i + 1 :],
            nside=nside,
            sky_bin_mapping=sky_bin_mapping,
        )

        # plot the sky fields on the sky map
        if field_pos is not None:
            skymap = plot_fields(
                time,
                current_radec=field_pos[i, :],
                completed_radec=field_pos[:i, :],
                future_radec=field_pos[i + 1 :, :],
                plot_zenith=False,
                plot_airmass=0,
                plot_galaxy=False,
                plot_moon=False,
                observer=None,
                skymap=skymap,
                current_kwargs={"edgecolor": "darkgreen", "c": "forestgreen", "s": 60},
                completed_kwargs={"edgecolor": "seagreen", "c": "none", "s": 60},
                future_kwargs={"edgecolor": "silver", "s": 60},
            )

        # save the figure
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
        required=True,
        help="Path to output gif file.",
    )
    parser.add_argument(
        "-s",
        "--schedule",
        type=str,
        required=True,
        help=(
            'Path to the schedule file, a csv file with keys "time" and some '
            'combination of "field_id", "policy_field_id", "bin_id", "policy_bin_id".'
        ),
    )
    parser.add_argument(
        "-t",
        "--plot-type",
        type=str,
        required=True,
        choices=["field", "bin", "fieldbin"],
        help=(
            'Whether to plot schedule of "field", "bin", or combined "fieldbin". '
            'Requires "(policy_)field_id" and/or "(policy_)bin_id" keys, respectively.'
        ),
    )
    parser.add_argument(
        "-c",
        "--compare",
        action="store_true",
        help=(
            "Switch to plot a comparison of two schedules, identified as with/without "
            '"policy_" in the schedule file.'
        ),
    )
    parser.add_argument(
        "-p",
        "--policy",
        action="store_true",
        help=(
            'Switch to plot "policy_" schedule keys as the primary schedule. When '
            'plotting fields, this selects "policy_field_id" instead of "field_id". '
            'When plotting bins, this uses "policy_bin_id" as the primary schedule, and'
            ' "bin_id" as the alternate schedule if using the compare option.'
        ),
    )
    parser.add_argument(
        "-f",
        "--fields",
        type=str,
        help=(
            "Path to field file, a json file that maps field_id to (ra, dec) in rad. "
            "Used for making field plots."
        ),
    )
    parser.add_argument(
        "-b",
        "--bins",
        type=str,
        help=(
            "Path to bin file, a json file that maps bin_id to (ra, dec) in rad. Not "
            "needed if nside provided, in which case this is used to verify grid "
            "reconstruction. Used for making bin plots."
        ),
    )
    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        help="nside used to make healpix sky bin for bin schedules",
    )
    args = parser.parse_args()

    # check argument compatibility
    if args.plot_type in ["field", "fieldbin"] and len(args.fields) == 0:
        raise ValueError("field file required to plot fields.")
    if args.plot_type in ["bin", "fieldbin"] and args.nside is None:
        raise ValueError("nside required to plot bins.")
    if args.plot_type == "field" and args.compare:
        raise NotImplementedError("Comparing fields for 2 schedules not implemented.")

    # load schedule file and check validity
    schedule = pd.read_csv(args.schedule)
    if "time" not in schedule.columns:
        raise KeyError('Missing "time" required to make requested plot.')
    if args.plot_type in ["field", "fieldbin"]:
        if args.policy and "policy_field_id" not in schedule.columns:
            raise KeyError('Missing "policy_field_id" required to make requested plot.')
        if not args.policy and "field_id" not in schedule.columns:
            raise KeyError('Missing "field_id" required to make requested plot.')
    if args.plot_type in ["bin", "fieldbin"]:
        if (args.compare or args.policy) and "policy_bin_id" not in schedule.columns:
            raise KeyError('Missing "policy_bin_id" required to make requested plot.')
        if (args.compare or not args.policy) and "bin_id" not in schedule.columns:
            raise KeyError('Missing "bin_id" required to make requested plot.')

    # load field and bin mappings
    field_id2pos = None
    if args.fields is not None:
        with open(args.fields) as f:
            field_id2pos = json.load(f)
    bin_id2pos = None
    if args.bins is not None:
        with open(args.bins) as f:
            bin_id2pos = json.load(f)

    # parse field, bin positions
    field_ids_1 = schedule.get("policy_field_id" if args.policy else "field_id", None)
    if field_ids_1 is None or field_id2pos is None:
        field_pos_1 = None
    else:
        field_pos_1 = np.asarray(
            [field_id2pos.get(str(fid), [None, None]) for fid in field_ids_1.values]
        )
    bin_ids_1 = schedule.get("policy_bin_id" if args.policy else "bin_id", None)
    bin_ids_2 = schedule.get("bin_id" if args.policy else "policy_bin_id", None)

    # call plotting functions
    if args.plot_type == "field":
        plot_fields_movie(
            outfile=args.outfile,
            times=schedule["time"].values,
            field_pos=field_pos_1,
        )
    else:
        plot_bins_movie(
            outfile=args.outfile,
            nside=args.nside,
            times=schedule["time"].values,
            idxs=bin_ids_1.values,
            alternate_idxs=bin_ids_2.values if args.compare else None,
            sky_bin_mapping=bin_id2pos,
            field_pos=field_pos_1 if args.plot_type == "fieldbin" else None,
        )
