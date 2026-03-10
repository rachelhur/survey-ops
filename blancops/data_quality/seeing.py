import numpy as np
from blancops.utils import units

# Hardcoded seeing conversion factors between bands
# Copied from obztak seeing.py, which in turn got them from Eric Neilsen
# Values are nominally (lambda_i / lambda_j)^(1/5), for effective band center lambda
# For more info, see https://doi.org/10.2172/1574836 Appendix C
_BAND_FACTORS = {
    "u": 0.86603,  # ( u (380nm) / i (780nm) ) ** (1/5)
    "g": 0.9067,  # ( g (480nm) / i (780nm) ) ** (1/5)
    "r": 0.9609,  # ( r (640nm) / i (780nm) ) ** (1/5)
    "i": 1.0,  # ( i (780nm) / i (780nm) ) ** (1/5)
    "z": 1.036,  # ( z (920nm) / i (780nm) ) ** (1/5)
    "Y": 1.0523,  # ( Y (990nm) / i (780nm) ) ** (1/5)
}

# Hardcoded contribution of DECam to the PSF
# From obztak seeing.py, which in turn got it from Eric Neilsen
_DECAM_FWHM = 0.5 * units.arcsec


def convert_seeing(
    seeing,
    to_band,
    to_el,
    from_band="i",
    from_el=90 * units.deg,
    instrument=_DECAM_FWHM,
):
    """
    Convert seeing from one band and elevation to another band and elevation assuming a
    Kolmogorov turbulence model, wherein seeing ~ airmass^(3/5) / wavelength^(1/5). For
    more info, see https://doi.org/10.2172/1574836 Appendix C.

    Arguments
    ---------
    seeing : float or list-like of float
        Seeing value to convert.
    to_band : str or list-like of str
        Target band for conversion ('u', 'g', 'r', 'i', 'z', or 'Y').
    to_el : float or list-like of float
        Target elevation for conversion (in radians).
    from_band : str or list-like of str ["i"]
        Original band of the seeing value ('u', 'g', 'r', 'i', 'z', or 'Y').
    from_el : float or list-like of float [90 * units.deg]
        Original elevation of the seeing value (in radians). Default is zenith.
    instrument : float [0.5 * units.arcsec]
        Component of seeing due to to the instrument, summed in quadrature with the
        atmospheric component. Default is nominal DECam contribution.

    Returns
    -------
    float or np.ndarray of float
        Converted seeing value.
    """
    # check for scalar inputs to preserve output shape
    scalar_input = np.all(
        [np.ndim(x) == 0 for x in [seeing, to_band, to_el, from_band, from_el]]
    )

    # remove instrument contribution in quadrature
    seeing = np.sqrt(np.asarray(seeing) ** 2 - instrument**2)

    # airmass conversion factor
    from_am = 1.0 / np.sin(from_el)
    to_am = 1.0 / np.sin(to_el)
    airmass_factor = (to_am / from_am) ** (3 / 5)

    # wavelength conversion factor
    to_band_factor = np.asarray(
        [_BAND_FACTORS[band] for band in np.atleast_1d(to_band)]
    )
    from_band_factor = np.asarray(
        [_BAND_FACTORS[band] for band in np.atleast_1d(from_band)]
    )
    band_factor = (to_band_factor / from_band_factor) ** (-1)

    # convert airmass and wavelength contributions and add back instrument contribution
    seeing = np.hypot(seeing * airmass_factor * band_factor, instrument)
    return np.squeeze(seeing) if scalar_input else seeing
