"""Run the three ilSN science steps with the public scientific functions.

Run from the source directory: PYTHONPATH=. python examples/quickstart.py
The small mass grid, Gaussian response, mock imaging and population priors
are prescribed tutorial choices, not a Roman survey-precision forecast.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ilsn.mass_grid import LensDesign, MassGrid
from ilsn.lens_bank import build_lens_bank
from ilsn.likelihood import (
    SourceSample,
    SourceLikelihood,
    SharedNuisance,
    magnification_log_likelihood,
)
from ilsn.observation import source_sample, td_selection, imaging_log_likelihood
from ilsn.population import MassPopulation, TargetSample, population_transfer
from ilsn.radial import RadialTransfer
from ilsn.response import ResponseCalibration
from ilsn.sampling import gaussian_nodes


def main(output_dir, seed=13):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    grid = MassGrid()
    population = MassPopulation(grid.base_probability, grid.scores)
    source_design = LensDesign(
        np.log10([1.5e11, 2e11, 2.5e11]),
        [0.7, 0.9, 1.1],
        0.3,
        1.2,
        radius_arcsec=[5.0, 7.0, 9.0],
    )
    source_bank = build_lens_bank(source_design, grid)
    # Twelve independent foreground/SN pairs share each design cell's
    # covariates; these are separate galaxies, with separate mass states.
    repeats = 12
    physics = {
        key: np.repeat(value, repeats, axis=0) for key, value in source_bank.items()
    }
    radius = np.repeat(source_design.radius_arcsec, repeats)
    size = np.repeat(source_design.re_arcsec, repeats)
    redshift = np.repeat(source_design.z_source, repeats)
    # No external LOS is an explicit tutorial assumption. Dust and stellar
    # microlensing follow the same public observation model as larger runs.
    sample = source_sample(
        np.zeros(len(redshift)),
        redshift,
        physics,
        radius_arcsec=radius,
        re_arcsec=size,
        los=None,
    )
    calibration = ResponseCalibration.gaussian(
        [0.8, 1.6],
        np.linspace(-2.0, 0.5, 41),
        scatter_mag=0.15,
        delivery_probability=1.0,
        zero_point_sigma=0.02,
        n_draws=16,
        rng=seed,
    )
    nuisance = SharedNuisance.from_calibration(
        calibration,
        [0.0, 0.5, 1.0],
        [0.25, 0.5, 0.25],
    )
    truth_eta = np.array([0.4, -0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    generator = SourceLikelihood(
        sample, calibration, nuisance, continuation="unit_photon"
    )
    sample, truth = generator.simulate(population, truth_eta, rng)
    likelihood = SourceLikelihood(
        sample, calibration, nuisance, continuation="unit_photon"
    )
    eta, eta_probability = gaussian_nodes(
        np.zeros(8), 0.7**2 * np.eye(8), n=64, seed=seed
    )
    source_logl = likelihood.log_likelihood(population, eta)

    # The first foreground's inner observation conditions both distributions.
    truth_state = int(truth["mass_state"][0])
    ring = source_bank["theta_e_arcsec"][0, truth_state]
    curvature = source_bank["xi_E"][0, truth_state]
    radial_imaging = imaging_log_likelihood(
        source_bank["theta_e_arcsec"][:1],
        source_bank["xi_E"][:1],
        [rng.normal(ring, 0.05 * ring)],
        [rng.normal(curvature, 0.15)],
        [0.05 * ring],
        [0.15],
    )[0]
    radial_sample = SourceSample(
        sample.residual_mag[:1],
        sample.redshift[:1],
        sample.photon_shift_mag[:1],
        sample.component_weight[:1],
        sample.extra_variance_mag2[:1],
        sample.dust_template_mag[:1],
    )
    radial = RadialTransfer(
        source_bank["s_E"][0],
        grid.base_probability,
        SourceLikelihood(
            radial_sample, calibration, nuisance, continuation="unit_photon"
        ),
        radial_imaging,
    )
    radial_forecast = radial.forecast(np.linspace(-3.0, 1.0, 401))

    target_design = LensDesign(
        np.log10([1.4e11, 1.8e11, 2.2e11, 2.6e11]),
        [0.65, 0.8, 1.0, 1.2],
        0.35,
        1.5,
    )
    target_bank = build_lens_bank(target_design, grid, source=False)
    admitted = (
        target_bank["physical_valid"]
        & target_bank["geometry_valid"]
        & target_bank["endpoint_valid"]
    )
    selection = td_selection(
        target_bank["theta_e_arcsec"],
        target_bank["caustic_area_arcsec2"],
        target_design.z_lens,
        target_design.z_source,
        admitted,
    )
    truth_delta = np.array([0.15, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    selected_probability = selection * population.probabilities(truth_eta + truth_delta)
    target_truth = np.array(
        [rng.choice(len(grid), p=p / p.sum()) for p in selected_probability]
    )
    rows = np.arange(len(target_design))
    true_ring = target_bank["theta_e_arcsec"][rows, target_truth]
    true_curvature = target_bank["xi_E"][rows, target_truth]
    target_imaging = imaging_log_likelihood(
        target_bank["theta_e_arcsec"],
        target_bank["xi_E"],
        rng.normal(true_ring, 0.05 * true_ring),
        rng.normal(true_curvature, 0.15),
        0.05 * true_ring,
        np.full(len(rows), 0.15),
    )
    targets = TargetSample(target_bank["s_E"], selection, target_imaging, admitted)
    delta, delta_probability = gaussian_nodes(
        np.zeros(8), 0.35**2 * np.eye(8), n=16, seed=seed + 1
    )
    fit = population_transfer(
        population,
        targets,
        eta,
        source_logl,
        eta_probability,
        delta,
        delta_probability,
    )
    report = dict(
        scope="Prescribed physical tutorial; not a Roman precision forecast",
        n_source=len(redshift),
        n_target=len(target_design),
        n_mass_states=len(grid),
        seed=seed,
        score_names=list(grid.score_names),
        source_radius_arcsec=source_design.radius_arcsec.tolist(),
        external_los="Explicitly absent",
        response="Complete-delivery Gaussian, scatter 0.15 mag",
        response_continuation="Explicit unit_photon",
        td_velocity_range_km_s=[150.0, 350.0],
        radial_prior_sd=radial.prior.standard_deviation,
        radial_realization_sd=radial.posterior().standard_deviation,
        radial_rms_posterior_sd=radial_forecast["rms_posterior_sd"],
        radial_predictive_integral=radial_forecast["density_integral"],
        target_mean_sd_before=fit.prior.catalogue_mean_sd,
        target_mean_sd_after=fit.posterior.catalogue_mean_sd,
        shared_calibration_node=int(truth["nuisance_index"]),
    )
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "posterior.npz",
        source_coordinates=eta,
        source_log_likelihood=source_logl,
        source_residual_mag=sample.residual_mag,
        source_radius_arcsec=radius,
        mass_base_probability=grid.base_probability,
        mass_scores=grid.scores,
        target_covariance_before=fit.prior.covariance,
        target_covariance_after=fit.posterior.covariance,
        target_mean_before=fit.prior.mean,
        target_mean_after=fit.posterior.mean,
        posterior_shared_node_probability=fit.posterior_node_probability,
        target_coordinates=fit.target_coordinates,
        target_selection=selection,
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    injected_mu = np.exp(source_bank["lnmu"][0, truth_state])
    brightness_mu = 10 ** (-0.4 * sample.residual_mag[0])
    mu = np.linspace(
        0.5 * min(injected_mu, brightness_mu),
        2.0 * max(injected_mu, brightness_mu),
        401,
    )
    logl = magnification_log_likelihood(sample.residual_mag[0], 1.2, mu, calibration)
    axes[0].plot(mu, np.exp(logl - logl.max()), color="#305f91")
    axes[0].axvline(injected_mu, color="#cf803e", ls="--", label="Injected macro value")
    axes[0].set(
        xlabel=r"Absolute magnification $\mu$",
        ylabel="Relative brightness likelihood",
        title="1. One SN brightness",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    endpoint = source_bank["s_E"][0]
    finite = np.isfinite(endpoint)
    bins = np.linspace(endpoint[finite].min() - 0.01, endpoint[finite].max() + 0.01, 15)
    for summary, color, label in (
        (radial.prior, "#305f91", "Imaging"),
        (radial.posterior(), "#cf803e", "Imaging + ilSN"),
    ):
        axes[1].hist(
            endpoint[finite],
            bins=bins,
            weights=summary.state_probability[finite],
            histtype="step",
            color=color,
            label=label,
        )
    axes[1].set(
        xlabel=r"Inner endpoint $s_E$",
        ylabel="Probability per bin",
        title="2. Radial transfer",
    )
    axes[1].legend(frameon=False, fontsize=9)
    for shift, summary, color, label in (
        (-0.07, fit.prior, "#305f91", "Target imaging"),
        (0.07, fit.posterior, "#cf803e", "+ Source population"),
    ):
        axes[2].errorbar(
            rows + shift,
            summary.mean,
            yerr=summary.standard_deviation,
            fmt="o",
            color=color,
            label=label,
        )
    axes[2].set(
        xlabel="Independent strong lens",
        ylabel=r"Target $s_E$ (mean $\pm$ SD)",
        title="3. Population transfer",
        xticks=rows,
        xticklabels=rows + 1,
    )
    axes[2].legend(frameon=False, fontsize=9, loc="lower left")
    fig.text(
        0.03,
        0.015,
        "First panel: dust and additional microlensing fixed. "
        "Gaussian tutorial response; no empirical precision claim.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    fig.savefig(output_dir / "three_steps.png", dpi=180)
    fig.savefig(output_dir / "three_steps.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quickstart"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    main(args.output_dir, args.seed)
