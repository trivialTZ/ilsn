"""Compare physical profiles with one common outer SN magnification.

This deliberately constructed family illustrates a radial information
limit. Lower stellar masses are compensated by positive broad finite cores
so that all models have the same absolute magnification at one fixed SN
angle. Some states additionally redistribute mass with a zero-total-mass
signed core pair. Reference mass and size remain fixed physical covariates;
each model determines its own Einstein ring and inner endpoint s_E.

An equal-weight discrete family and a complete-delivery Gaussian response
are prescribed for illustration. No measured population, survey calibration,
imaging likelihood, or precision claim is supplied. Because all states have
the same outer brightness, RadialTransfer correctly retains their inner
uncertainty after the SN. The geometry check assumes circular, decreasing
density; positivity and monotonicity are sampled, not globally certified.

Run from the source directory after installing its dependencies::

    PYTHONPATH=. python examples/radial_models.py --output-dir outputs/radial_models
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.optimize import brentq

from ilsn.geometry import radial_caustic_radius, single_image_mask
from ilsn.lens_model import CompositeProfile, GalaxyProfile
from ilsn.likelihood import SourceLikelihood, SourceSample
from ilsn.radial import RadialTransfer
from ilsn.response import ResponseCalibration


def matched_profiles():
    """Construct four admitted profiles with common outer magnification.

    Stellar mass varies while halo M200 and concentration remain fixed.
    Broad-core eta is solved on the nonnegative branch to match the
    fiducial magnification at theta_SN=3 theta_E(fiducial). The SN angle
    is held fixed across states, rather than rescaled by each derived ring.

    :return: Model lenses, parameter/diagnostic rows, and reference geometry.
    :rtype: tuple
    """
    reference_mass = 1e11
    fiducial = GalaxyProfile(
        z_l=0.3,
        re_arcsec=1.0,
        stellar_mass_msun=reference_mass,
        halo_mass_msun=1e13,
        concentration=7.0,
    )
    reference_re = fiducial.re_kpc
    z_source = 1.0
    fiducial_lens = fiducial.lens(z_source)
    theta_e_fiducial = fiducial_lens.theta_E()
    theta_sn = 3 * theta_e_fiducial
    common_mu = float(abs(fiducial_lens.mu(theta_sn)))
    configurations = (
        ("Reference", 1.0, 5.0, 0.0),
        (r"0.8 $M_\star$, $q=3$", 0.8, 3.0, 0.05),
        (r"0.8 $M_\star$, $q=10$", 0.8, 10.0, 0.05),
        (r"0.6 $M_\star$, $q=10$", 0.6, 10.0, 0.0),
    )
    physical_radius = np.geomspace(0.01, 1000, 513)
    lenses, rows = [], []
    for label, stellar_factor, q, compensation in configurations:
        galaxy = GalaxyProfile(
            z_l=fiducial.z_l,
            re_arcsec=fiducial.re_arcsec,
            stellar_mass_msun=stellar_factor * reference_mass,
            halo_mass_msun=fiducial.halo_mass_msun,
            concentration=fiducial.concentration,
        )

        def profile(eta):
            return CompositeProfile.from_reference(
                galaxy,
                reference_mass,
                reference_re,
                eta,
                q,
                compensation=compensation,
                reference_q=5.0,
                compensation_radii=(1.0, 3.0),
            )

        def mismatch(eta):
            return float(profile(eta).lens(z_source).mu(theta_sn)) - common_mu

        eta = 0.0 if stellar_factor == 1.0 else brentq(mismatch, 0.0, 0.2, xtol=1e-13)
        physical = profile(eta)
        support = physical.physical_support(physical_radius)
        if not support["sampled_valid"]:
            raise ValueError(
                f"Constructed state fails sampled physical support: {label}"
            )
        lens = physical.lens(z_source)
        # These finite-cusp stars/gNFW/cores have alpha(0+)=0. Resolve all
        # bracketed radial critical points inside each model's own ring.
        geometry_grid = np.geomspace(
            lens.minimum_radius_arcsec * 1.001, lens.theta_E(), 2049
        )
        caustic = radial_caustic_radius(
            lens, geometry_grid, central_deflection_limit=0.0
        )
        single_image = bool(single_image_mask(lens, theta_sn, caustic))
        if not single_image:
            raise ValueError(
                f"The fixed SN position is not a single outer image: {label}"
            )
        rows.append(
            dict(
                label=label,
                stellar_mass_factor=stellar_factor,
                stellar_mass_msun=galaxy.stellar_mass_msun,
                eta_core=eta,
                core_q=q,
                compensation=compensation,
                signed_core_mass_msun=physical.total_core_mass_msun,
                theta_e_arcsec=lens.theta_E(),
                s_e=lens.s_E(),
                xi_e=lens.xi_E(),
                theta_sn_arcsec=theta_sn,
                absolute_mu_sn=float(abs(lens.mu(theta_sn))),
                source_radius_arcsec=float(theta_sn - lens.deflection(theta_sn)),
                caustic_radius_arcsec=caustic,
                single_image=single_image,
                sampled_physical_valid=support["sampled_valid"],
            )
        )
        lenses.append(lens)
    reference = dict(
        reference_mass_msun=reference_mass,
        reference_re_kpc=reference_re,
        lens_redshift=fiducial.z_l,
        source_redshift=z_source,
        theta_e_fiducial_arcsec=theta_e_fiducial,
        fixed_theta_sn_arcsec=theta_sn,
        absolute_mu_sn=common_mu,
        physical_grid_min_kpc=float(physical_radius.min()),
        physical_grid_max_kpc=float(physical_radius.max()),
        physical_grid_count=len(physical_radius),
    )
    return lenses, rows, reference


def radial_information(rows, source_redshift):
    """Evaluate unchanged inner uncertainty along the constructed null family.

    :param rows: Parameter records returned by matched_profiles.
    :param source_redshift: Response-conditioning source redshift.
    :return: Selected prior/posterior endpoint summaries and predictive check.
    :rtype: dict
    """
    mu = np.array([row["absolute_mu_sn"] for row in rows])
    shift = -2.5 * np.log10(mu)
    endpoint = np.array([row["s_e"] for row in rows])
    scatter_mag = 0.15
    calibration = ResponseCalibration.gaussian(
        [0.5, 1.5],
        [-1.5, 0.5],
        scatter_mag=scatter_mag,
        delivery_probability=1.0,
        zero_point_sigma=0.0,
        n_draws=1,
        rng=0,
    )
    observed_residual = float(shift[0])
    sample = SourceSample(
        [observed_residual], [source_redshift], shift[None, :], component_weight=1.0
    )
    radial = RadialTransfer(
        endpoint, np.ones(len(rows)), SourceLikelihood(sample, calibration)
    )
    prior, posterior = radial.prior, radial.posterior()
    forecast = radial.forecast(
        np.linspace(
            observed_residual - 7 * scatter_mag,
            observed_residual + 7 * scatter_mag,
            501,
        )
    )
    if not np.allclose(
        prior.state_probability, posterior.state_probability, atol=1e-10, rtol=0
    ):
        raise ValueError(
            "Equal outer magnifications must retain the radial state probabilities"
        )
    return dict(
        prescribed_scatter_mag=scatter_mag,
        observed_residual_mag=observed_residual,
        prior_s_e_mean=prior.mean,
        prior_s_e_sd=prior.standard_deviation,
        posterior_s_e_mean=posterior.mean,
        posterior_s_e_sd=posterior.standard_deviation,
        predictive_integral=forecast["density_integral"],
        rms_posterior_s_e_sd=forecast["rms_posterior_sd"],
        prior_state_probability=prior.state_probability.tolist(),
        posterior_state_probability=posterior.state_probability.tolist(),
    )


def plot_profiles(lenses, rows, reference, information, output_dir):
    """Save density, magnification, and inner-endpoint comparison panels.

    :param lenses: Matched physical SphericalLens objects.
    :param rows: Model parameters and geometry diagnostics.
    :param reference: Fixed physical covariates and fiducial angle.
    :param information: Prescribed RadialTransfer result.
    :param output_dir: Directory for PNG and PDF figures.
    :return: Saved figure paths.
    :rtype: list[pathlib.Path]
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ("#165DAD", "#D46B20", "#D46B20", "#15384F")
    styles = ("-", "-", "--", "-")
    markers = ("o", "o", "x", "s")
    theta_e = reference["theta_e_fiducial_arcsec"]
    density_radius = np.geomspace(0.15, 8.0, 401)
    outer_radius = np.geomspace(1.3, 8.0, 401)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6))
    fig.subplots_adjust(left=0.065, right=0.97, top=0.81, bottom=0.28, wspace=0.35)
    for lens, row, color, style, marker in zip(lenses, rows, colors, styles, markers):
        axes[0].plot(
            density_radius,
            lens.kappa(density_radius * theta_e),
            color=color,
            linestyle=style,
            label=row["label"],
            lw=1.8,
        )
        axes[1].plot(
            outer_radius,
            abs(lens.mu(outer_radius * theta_e)),
            color=color,
            linestyle=style,
            lw=1.8,
        )
        axes[2].scatter(
            row["absolute_mu_sn"],
            row["s_e"],
            color=color,
            marker=marker,
            s=58,
            zorder=3,
        )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel=r"$\theta/\theta_{\rm E,fid}$",
        ylabel=r"$\kappa(\theta)$",
        title="Projected density",
    )
    axes[0].set_xticks([0.2, 0.5, 1, 2, 5], labels=["0.2", "0.5", "1", "2", "5"])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[1].set(
        xscale="log",
        xlabel=r"$\theta/\theta_{\rm E,fid}$",
        ylabel=r"$|\mu(\theta)|$",
        title="Outer magnification",
    )
    axes[1].set_xticks([1.5, 2, 3, 5, 8], labels=["1.5", "2", "3", "5", "8"])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].axvline(3, color="0.65", lw=1, linestyle=":")
    axes[1].text(3.15, 3.2, "Fixed SN angle", color="0.35", fontsize=9, rotation=90)
    common_mu = reference["absolute_mu_sn"]
    axes[2].axvline(common_mu, color="0.7", lw=1, linestyle=":")
    axes[2].set(
        xlim=(common_mu * 0.97, common_mu * 1.03),
        xlabel=r"$|\mu(\theta_{\rm SN})|$",
        ylabel=r"$s_{\rm E}=1-\kappa(\theta_{\rm E})$",
        title="Each model's inner endpoint",
    )
    for axis in axes:
        axis.grid(alpha=0.13)
    fig.suptitle("One outer brightness, several inner profiles", y=0.96, fontsize=15)
    fig.text(
        0.065,
        0.15,
        r"All states match the brightness at $\theta_{\rm SN}=3\theta_{\rm E,fid}$; "
        r"each uses its own $\theta_{\rm E}$ for $s_{\rm E}$.",
        fontsize=10,
    )
    fig.text(
        0.065,
        0.08,
        r"Prescribed equal-weight family: prior SD($s_{\rm E}$) = "
        f"{information['prior_s_e_sd']:.4f}; "
        r"SN posterior SD($s_{\rm E}$) = "
        f"{information['posterior_s_e_sd']:.4f}. "
        "Illustration, with no empirical calibration.",
        fontsize=9,
        color="0.3",
    )
    paths = [output_dir / "radial_models.png", output_dir / "radial_models.pdf"]
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return paths


def main():
    """Write reproducible figure and model diagnostics to a chosen directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("radial_models_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lenses, rows, reference = matched_profiles()
    information = radial_information(rows, reference["source_redshift"])
    figure_paths = plot_profiles(lenses, rows, reference, information, args.output_dir)
    with (args.output_dir / "radial_models.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        reference=reference,
        radial_information=information,
        assumptions=[
            "Prescribed four-state family, with equal prior probability and no imaging likelihood.",
            "Independent stellar and halo masses; finite core mass adds to M200 without re-budgeting it.",
            "Positive broad-core amplitude; signed compact redistribution has zero total mass.",
            "All states share a fixed angular SN radius from the fiducial lens, not their own Einstein rings.",
            "Sampled positive/decreasing density on 0.01--1000 kpc; no global or distribution-function certificate.",
            "Circular single-image eligibility checks the outer minimum and excludes opposite-side images.",
            "Complete delivery and 0.15-mag Gaussian scatter are imposed demonstration assumptions.",
            "This example illustrates a radial information limit, not a Roman population or precision forecast.",
        ],
    )
    (args.output_dir / "radial_models.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Saved {len(rows)} physical profiles to {args.output_dir}")
    print(
        f"Fixed theta_SN = {reference['fixed_theta_sn_arcsec']:.4f} arcsec; "
        f"common absolute mu = {reference['absolute_mu_sn']:.6f}"
    )
    print(
        f"Prior / SN posterior SD(s_E): {information['prior_s_e_sd']:.6f} / "
        f"{information['posterior_s_e_sd']:.6f}"
    )
    print(f"Predictive density integral: {information['predictive_integral']:.12f}")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
