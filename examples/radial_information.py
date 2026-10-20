"""Locate the SN angles that inform the Einstein-radius endpoint.

One standardized SN is placed at a fixed observed angle behind a TD-like
foreground galaxy. For each angle we forecast how much one brightness
measurement narrows s_E=1-kappa(theta_E), averaging posterior variance over
the selected predictive residual distribution (``RadialTransfer.forecast``).
The scan is repeated for three radial families and with or without a
compressed inner-imaging observation:

* ``full``: broad cores at q=3, 5, 10 and compensated redistribution;
* ``fixed_shape``: one core scale q=5 and no redistribution;
* ``no_core``: no broad core or redistribution.

Angles are multiples of the fiducial state's Einstein radius; every state
keeps its own ring and single-image admission. The response is the bundled
simulated Roman TRAIN20 point curve at z=1.2 with unit-photon continuation,
integrated over shared dust and residual-offset nodes. Imaging is centered
without noise on the fiducial state's theta_E and xi_E, so the result is a
conditional design comparison for this family and measure, not a Roman
survey forecast or an empirical radial prior.

Run from the source directory after installing its dependencies::

    PYTHONPATH=. python examples/radial_information.py --output-dir outputs/radial_information
    PYTHONPATH=. python examples/radial_information.py --grid standard

The 486-state tutorial grid takes about two minutes; the 2,430-state standard
grid adds halo-cusp and finer core-amplitude nodes and takes about ten.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ilsn.lens_bank import build_lens_bank
from ilsn.likelihood import SharedNuisance, SourceLikelihood
from ilsn.mass_grid import LensDesign, MassGrid, PhysicalRelations
from ilsn.observation import imaging_log_likelihood, source_sample
from ilsn.radial import RadialTransfer
from ilsn.response import ResponseCalibration

ROOT = Path(__file__).resolve().parents[1]
GRIDS = dict(
    tutorial=dict(
        mass_multipliers=(0.8, 1.0, 1.25),
        halo_nodes=(-1.0, 0.0, 1.0),
        concentration_nodes=(0.0,),
        gradients=(-0.3, 0.0),
        halo_slopes=(1.0,),
        eta_nodes=(-0.1, 0.0, 0.1),
        q_nodes=(3.0, 5.0, 10.0),
        compensation_nodes=(-0.05, 0.0, 0.05),
    ),
    standard=dict(
        mass_multipliers=(0.8, 1.0, 1.25),
        halo_nodes=(-1.0, 0.0, 1.0),
        concentration_nodes=(0.0,),
        gradients=(-0.3, 0.0),
        halo_slopes=(0.7, 1.0, 1.3),
        eta_nodes=(-0.1, -0.05, 0.0, 0.05, 0.1),
        q_nodes=(3.0, 5.0, 10.0),
        compensation_nodes=(-0.05, 0.0, 0.05),
    ),
)
FAMILIES = dict(
    full="q=3, 5, 10 and compensated redistribution",
    fixed_shape="q=5, no redistribution",
    no_core="no broad core or redistribution",
)
METRICS = (
    "prior_sd",
    "rms_posterior_sd",
    "sd_ratio",
    "information_gain",
    "predictive_integral",
    "coarse_grid_relative_change",
)
ADMISSION = (
    "physical_valid",
    "geometry_valid",
    "projection_valid",
    "endpoint_valid",
    "single_image",
)


def family_masks(grid):
    """Boolean state masks for the three radial families.

    :param grid: MassGrid containing q=5 and zero core/redistribution nodes.
    :return: Dictionary of masks with shape (state,).
    """
    fixed = np.isclose(grid.core_q, 5.0) & np.isclose(grid.compensation, 0.0)
    masks = dict(
        full=np.ones(len(grid), bool),
        fixed_shape=fixed,
        no_core=fixed & np.isclose(grid.eta_core, 0.0),
    )
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("Grid needs q=5, zero redistribution and zero core nodes")
    return masks


def fiducial_state(grid):
    """Index of the central state: no core, unit mass, mean halo and cusp."""
    match = (
        np.isclose(grid.mass_multiplier, 1.0)
        & np.isclose(grid.halo_residual, 0.0)
        & np.isclose(grid.concentration_residual, 0.0)
        & np.isclose(grid.stellar_gradient, 0.0)
        & np.isclose(grid.halo_slope, 1.0)
        & np.isclose(grid.eta_core, 0.0)
        & np.isclose(grid.core_q, 5.0)
        & np.isclose(grid.compensation, 0.0)
    )
    if match.sum() != 1:
        raise ValueError("Grid must contain exactly one central fiducial state")
    return int(np.flatnonzero(match)[0])


def scan(
    grid,
    multiples,
    *,
    log_design_mass=11.3,
    re_arcsec=0.9,
    z_lens=0.3,
    z_source=1.2,
    theta_error_fraction=0.05,
    xi_error=0.15,
    residual_points=401,
):
    """Forecast the s_E update from one SN at each angle, family and inner datum.

    :param grid: MassGrid for the source foreground.
    :param multiples: SN angles in units of the fiducial Einstein radius.
    :param theta_error_fraction: Fractional theta_E error of the mock imaging.
    :param xi_error: Absolute xi_E error of the mock imaging.
    :param residual_points: Predictive residual grid size; a half-resolution
        grid is evaluated as a convergence check.
    :return: Result rows and the fiducial geometry.
    """
    relations = PhysicalRelations()
    fid = fiducial_state(grid)
    ring_design = LensDesign(log_design_mass, re_arcsec, z_lens, z_source)
    ring = build_lens_bank(
        ring_design, grid, relations, source=False, state_slice=[fid]
    )
    theta_fid, xi_fid = ring["theta_e_arcsec"][0, 0], ring["xi_E"][0, 0]
    if not np.isfinite(theta_fid):
        raise ValueError("Fiducial state has no Einstein ring")
    multiples = np.asarray(multiples, float)
    design = LensDesign(
        log_design_mass,
        re_arcsec,
        z_lens,
        z_source,
        radius_arcsec=multiples * theta_fid,
    )
    bank = build_lens_bank(design, grid, relations)
    if bank["support_unresolved"].any():
        raise ValueError("Resolve material support before inference")
    endpoint = np.where(bank["endpoint_valid"][0], bank["s_E"][0], 0.0)
    log_imaging = imaging_log_likelihood(
        bank["theta_e_arcsec"][:1],
        bank["xi_E"][:1],
        [theta_fid],
        [xi_fid],
        [theta_error_fraction * theta_fid],
        [xi_error],
    )[0]
    response = ResponseCalibration.load(ROOT / "data" / "roman_response.npz")
    nuisance = SharedNuisance(
        response_index=np.full(6, -1),
        probability=np.repeat([0.25, 0.5, 0.25], 2) * np.tile([0.5, 0.5], 3),
        dust_amplitude=np.tile([0.0, 0.5], 3),
        residual_offset_mag=np.repeat([-0.02, 0.0, 0.02], 2),
    )
    masks = family_masks(grid)
    rows = []
    for j, multiple in enumerate(multiples):
        physics = {key: value[j : j + 1] for key, value in bank.items()}
        admitted = np.logical_and.reduce([bank[name][j] for name in ADMISSION])
        sample = source_sample(
            [0.0],
            design.z_source[j : j + 1],
            physics,
            radius_arcsec=design.radius_arcsec[j : j + 1],
            re_arcsec=design.re_arcsec[j : j + 1],
            los=None,
        )
        likelihood = SourceLikelihood(
            sample, response, nuisance, continuation="unit_photon"
        )
        for family, keep in masks.items():
            states = admitted & keep
            for inner, logi in (("none", 0.0), ("imaging", log_imaging)):
                row = dict(
                    theta_sn_over_theta_e_fid=float(multiple),
                    theta_sn_arcsec=float(design.radius_arcsec[j]),
                    family=family,
                    inner_data=inner,
                    admitted_states=int(states.sum()),
                    admitted_fraction=float(
                        grid.base_probability @ states / (grid.base_probability @ keep)
                    ),
                )
                if states.sum() < 2:
                    rows.append(row | dict.fromkeys(METRICS, np.nan))
                    continue
                transfer = RadialTransfer(
                    endpoint,
                    grid.base_probability,
                    likelihood,
                    np.where(states, logi, -np.inf),
                )
                shift = sample.photon_shift_mag[0][states]
                # Pad by >8 selected-scatter widths including dust and offsets.
                residuals = np.linspace(
                    shift.min() - 2.0, shift.max() + 2.0, residual_points
                )
                fine = transfer.forecast(residuals)
                coarse = transfer.forecast(residuals[::2])
                prior_sd, rms = fine["prior_sd"], fine["rms_posterior_sd"]
                row.update(
                    prior_sd=prior_sd,
                    rms_posterior_sd=rms,
                    sd_ratio=rms / prior_sd if prior_sd > 0 else np.nan,
                    information_gain=(
                        1 / rms**2 - 1 / prior_sd**2 if rms > 0 else np.nan
                    ),
                    predictive_integral=fine["density_integral"],
                    coarse_grid_relative_change=abs(
                        coarse["rms_posterior_sd"] / rms - 1
                    ),
                )
                rows.append(row)
    fiducial = dict(
        log_design_mass=log_design_mass,
        re_arcsec=re_arcsec,
        z_lens=z_lens,
        z_source=z_source,
        theta_e_fid_arcsec=float(theta_fid),
        xi_e_fid=float(xi_fid),
        s_e_fid=float(ring["s_E"][0, 0]),
        theta_error_arcsec=float(theta_error_fraction * theta_fid),
        xi_error=xi_error,
        n_states=len(grid),
    )
    return rows, fiducial


def plot(rows, fiducial, output_dir):
    """Save admission and expected-SD-ratio panels.

    :return: Saved figure paths.
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
    colors = dict(full="#165DAD", fixed_shape="#D46B20", no_core="#15384F")
    styles = dict(full="-", fixed_shape="--", no_core=":")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.2, wspace=0.32)
    for family in FAMILIES:
        picked = [r for r in rows if r["family"] == family]
        for inner, axis in (("none", axes[1]), ("imaging", axes[2])):
            chosen = [r for r in picked if r["inner_data"] == inner]
            axis.plot(
                [r["theta_sn_over_theta_e_fid"] for r in chosen],
                [r["sd_ratio"] for r in chosen],
                color=colors[family],
                linestyle=styles[family],
                marker="o",
                ms=3.5,
                label=FAMILIES[family],
            )
        unique = [r for r in picked if r["inner_data"] == "none"]
        axes[0].plot(
            [r["theta_sn_over_theta_e_fid"] for r in unique],
            [r["admitted_fraction"] for r in unique],
            color=colors[family],
            linestyle=styles[family],
            marker="o",
            ms=3.5,
            label=FAMILIES[family],
        )
    axes[0].set(
        xscale="log",
        xlabel=r"$\theta_{\rm SN}/\theta_{\rm E,fid}$",
        ylabel="Admitted fraction of family base mass",
        title="Single-image, physical states",
        ylim=(0, 1.02),
    )
    axes[1].set(
        xscale="log",
        xlabel=r"$\theta_{\rm SN}/\theta_{\rm E,fid}$",
        ylabel=r"Expected SD($s_{\rm E}$) / prior SD",
        title="One SN, no inner data",
        ylim=(0, 1.05),
    )
    axes[2].set(
        xscale="log",
        xlabel=r"$\theta_{\rm SN}/\theta_{\rm E,fid}$",
        ylabel=r"Expected SD($s_{\rm E}$) / prior SD",
        title="One SN after compressed imaging",
        ylim=(0, 1.05),
    )
    ticks = [1.5, 2, 3, 4, 6, 8]
    for axis in axes:
        axis.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
        axis.xaxis.set_minor_formatter(plt.NullFormatter())
        axis.grid(alpha=0.13)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Which SN angles inform the Einstein-radius endpoint?", fontsize=14)
    fig.text(
        0.06,
        0.04,
        rf"Fiducial $\theta_{{\rm E}}$={fiducial['theta_e_fid_arcsec']:.2f} arcsec; "
        rf"imaging $\sigma_\theta$={fiducial['theta_error_arcsec']:.3f} arcsec, "
        rf"$\sigma_\xi$={fiducial['xi_error']:g}. Roman TRAIN20 response at "
        r"$z$=1.2; prescribed family and measure, no survey-precision claim.",
        fontsize=9,
        color="0.3",
    )
    paths = [
        output_dir / "radial_information.png",
        output_dir / "radial_information.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return paths


def main():
    """Write the radial-information scan, its diagnostics and a figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/radial_information")
    )
    parser.add_argument("--grid", choices=sorted(GRIDS), default="tutorial")
    parser.add_argument(
        "--multiples",
        type=float,
        nargs="+",
        default=[1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0],
        help="SN angles in units of the fiducial Einstein radius",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = MassGrid(**GRIDS[args.grid])
    rows, fiducial = scan(grid, args.multiples)
    with (args.output_dir / "radial_information.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = plot(rows, fiducial, args.output_dir)
    summary = dict(
        scope="Conditional radial-information design scan; not a Roman forecast",
        grid=args.grid,
        grid_axes=GRIDS[args.grid],
        fiducial=fiducial,
        families=FAMILIES,
        rows=[
            {
                k: None if isinstance(v, float) and np.isnan(v) else v
                for k, v in r.items()
            }
            for r in rows
        ],
        assumptions=[
            "Spherical Sersic stars, gNFW halo and signed finite cores; certified material support.",
            "Circular single-image admission at the fixed observed SN angle; each state uses its own ring.",
            "Simulated Roman TRAIN20 point response at z=1.2, unit-photon continuation, selected delivery.",
            "Shared dust amplitudes 0/0.5 and residual offsets -0.02/0/+0.02 mag; no external line of sight.",
            "Noise-free compressed imaging centered on the fiducial theta_E and xi_E.",
            "Restricting a family renormalizes the remaining base measure; compare SDs and ratios.",
        ],
    )
    (args.output_dir / "radial_information.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"Fiducial theta_E = {fiducial['theta_e_fid_arcsec']:.4f} arcsec; "
        f"{fiducial['n_states']} states"
    )
    print(
        f"{'theta/thE':>9s} {'family':>12s} {'inner':>8s} {'admitted':>8s} "
        f"{'prior SD':>9s} {'exp. SD':>9s} {'ratio':>6s}"
    )
    for row in rows:
        print(
            f"{row['theta_sn_over_theta_e_fid']:9.2f} {row['family']:>12s} "
            f"{row['inner_data']:>8s} {row['admitted_fraction']:8.3f} "
            f"{row['prior_sd']:9.5f} {row['rms_posterior_sd']:9.5f} "
            f"{row['sd_ratio']:6.3f}"
        )
    worst = np.nanmax([r["coarse_grid_relative_change"] for r in rows])
    print(f"Largest half-resolution residual-grid change: {worst:.2e}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
