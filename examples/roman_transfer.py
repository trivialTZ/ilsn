"""Recompute conditional population transfer from saved Roman simulation arrays.

The bundled input is a complete reduced calculation: two SN foregrounds,
ten independent strong-lens targets, 1,620 mass states and 64 joint source
nodes in the actual eight population coordinates. It is not a survey forecast.
No fitted posterior is read from disk. See data/README.rst for input assumptions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from ilsn.inference import source_posterior
from ilsn.observation import source_sample, td_selection, imaging_log_likelihood
from ilsn.population import MassPopulation, TargetSample, target_readout
from ilsn.response import ResponseCalibration
from ilsn.sampling import gaussian_nodes, receiving_nodes


def source_blocks(data, realization, block_size):
    """Construct one block's local photon/LOS arrays at a time."""
    los = {
        key: data[f"los_{key}"] for key in ("kappa", "gamma", "cos2phi", "probability")
    }
    residual = data["source_residual_mag"][realization]
    for start in range(0, len(residual), block_size):
        rows = slice(start, start + block_size)
        physics = {
            key: data[f"source_{key}"][rows]
            for key in (
                "kappa",
                "mean_kappa",
                "kappa_star",
                "physical_valid",
                "single_image",
                "geometry_valid",
                "projection_valid",
            )
        }
        sample = source_sample(
            residual[rows],
            data["source_redshift"][rows],
            physics,
            radius_arcsec=data["source_radius_arcsec"][rows],
            re_arcsec=data["source_re_arcsec"][rows],
            los=los,
            los_cell=data["source_los_cell"][rows],
        )
        yield sample, data["source_offset_mag"][:, rows]


def run(
    data_path,
    response_path,
    output_dir,
    *,
    realization=0,
    nodes=1024,
    seed=31,
    contrast_scale=1.0,
    block_size=1,
    continuation="unit_photon",
):
    """Run fresh source inference and the same target observations before/after."""
    if block_size < 1 or not np.isfinite(contrast_scale) or contrast_scale < 0:
        raise ValueError("Use a positive block size and nonnegative contrast scale.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if not 0 <= realization < len(data["source_residual_mag"]):
        raise ValueError("The requested simulation realization is absent.")
    response = ResponseCalibration.load(response_path)
    source_population = MassPopulation(
        data["source_base_probability"], data["source_scores"]
    )
    eta = data["source_eta"]
    fit = source_posterior(
        source_blocks(data, realization, block_size),
        source_population,
        response,
        eta,
        data["source_response_index"],
        data["source_log_prior_weight"],
        data["dust_amplitude"],
        data["dust_probability"],
        continuation=continuation,
    )

    target_population = MassPopulation(
        data["target_base_probability"], data["target_scores"]
    )
    admitted = (
        data["target_physical_valid"]
        & data["target_geometry_valid"]
        & data["target_endpoint_valid"]
    )
    selection = td_selection(
        data["target_theta_e_arcsec"],
        data["target_caustic_area_arcsec2"],
        data["target_z_lens"],
        data["target_z_source"],
        admitted,
    )
    imaging = imaging_log_likelihood(
        data["target_theta_e_arcsec"],
        data["target_xi_E"],
        data["observed_theta_e"],
        data["observed_xi_e"],
        data["sigma_theta_e"],
        data["sigma_xi_e"],
    )
    targets = TargetSample(data["target_s_E"], selection, imaging, admitted)
    contrast_covariance = contrast_scale**2 * data["contrast_covariance"]
    prior_nodes, prior_weights = gaussian_nodes(
        np.zeros(eta.shape[1]),
        data["source_prior_covariance"] + contrast_covariance,
        n=nodes,
        seed=seed,
    )
    posterior_nodes, posterior_weights = receiving_nodes(
        eta,
        fit["probability"],
        contrast_covariance,
        n=nodes,
        seed=seed,
    )
    before, pb = target_readout(target_population, targets, prior_nodes, prior_weights)
    after, pa = target_readout(
        target_population, targets, posterior_nodes, posterior_weights
    )
    # A coarse importance rule can represent its posterior much better than
    # its prior. Keep a same-rule comparison so this difference is not called
    # an SN gain. The separate Gaussian reference retains the scientific prior.
    logw = data["source_log_prior_weight"]
    source_prior_probability = np.exp(logw - logsumexp(logw))
    matched_nodes, matched_weights = receiving_nodes(
        eta,
        source_prior_probability,
        contrast_covariance,
        n=nodes,
        seed=seed,
    )
    matched_before, pm = target_readout(
        target_population, targets, matched_nodes, matched_weights
    )
    report = dict(
        scope="Reduced Roman simulation calculation; not a survey-precision forecast",
        n_source=fit["n_source"],
        n_target=len(before.mean),
        source_mass_states=len(source_population.base_probability),
        target_mass_states=len(target_population.base_probability),
        source_joint_nodes=len(eta),
        receiving_nodes=nodes,
        realization=realization,
        seed=seed,
        contrast_scale=contrast_scale,
        response_continuation=continuation,
        source_effective_nodes=float(1 / (fit["probability"] ** 2).sum()),
        source_prior_effective_nodes=float(1 / (source_prior_probability**2).sum()),
        target_mean_before=before.catalogue_mean,
        target_mean_after=after.catalogue_mean,
        target_mean_sd_before=before.catalogue_mean_sd,
        target_mean_sd_after=after.catalogue_mean_sd,
        matched_rule_mean_before=matched_before.catalogue_mean,
        matched_rule_sd_before=matched_before.catalogue_mean_sd,
        matched_rule_width_ratio=after.catalogue_mean_sd
        / matched_before.catalogue_mean_sd,
        gaussian_reference_width_ratio=after.catalogue_mean_sd
        / before.catalogue_mean_sd,
        no_sn_rule_sd_difference=matched_before.catalogue_mean_sd
        - before.catalogue_mean_sd,
        integration_note=(
            "Matched-rule ratio isolates the SN update on the supplied finite rule. "
            "The Gaussian-reference ratio includes integration mismatch; it is not an SN-only gain. "
            "The no-SN difference is a diagnostic, not an error bound."
        ),
        assumptions=[
            "Spherical stars+gNFW with signed finite core; q=5 and no redistribution in this input",
            "Frozen eight-score exponential population; fixed assigned mass/size/redshift covariates",
            "Paired external convergence/shear and local Jacobian; prescribed post-delivery microlensing",
            "Common empirical response, control offsets and dust; selected fixed-size source likelihood",
            "TD-like caustic-area selection with 150-350 km/s SIS proxy and compressed mock imaging",
            "One common source-target contrast, independent of the source likelihood",
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "posterior.npz",
        source_coordinates=eta,
        source_probability=fit["probability"],
        source_conditional_log_likelihood=fit["conditional_log_likelihood"],
        source_log_likelihood=fit["log_likelihood"],
        target_coordinates_before=prior_nodes,
        target_coordinates_after=posterior_nodes,
        target_node_probability_before=pb,
        target_node_probability_after=pa,
        target_coordinates_matched_before=matched_nodes,
        target_node_probability_matched_before=pm,
        source_prior_probability=source_prior_probability,
        target_mean_matched_before=matched_before.mean,
        target_covariance_matched_before=matched_before.covariance,
        target_mean_before=before.mean,
        target_mean_after=after.mean,
        target_covariance_before=before.covariance,
        target_covariance_after=after.covariance,
        target_shared_covariance_before=before.shared_covariance,
        target_shared_covariance_after=after.shared_covariance,
        target_individual_variance_before=before.individual_variance,
        target_individual_variance_after=after.individual_variance,
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/roman_transfer.npz")
    parser.add_argument(
        "--response", type=Path, default=root / "data/roman_response.npz"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/roman_transfer")
    )
    parser.add_argument("--realization", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--contrast-scale", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument(
        "--continuation", choices=("raise", "unit_photon"), default="unit_photon"
    )
    args = parser.parse_args()
    run(
        args.data,
        args.response,
        args.output_dir,
        realization=args.realization,
        nodes=args.nodes,
        seed=args.seed,
        contrast_scale=args.contrast_scale,
        block_size=args.block_size,
        continuation=args.continuation,
    )
