"""Trace which population directions carry ilSN information to TD lenses.

This is a linearized (Fisher) companion to ``roman_transfer.py``. At the
reference population eta=0 it computes, for a catalogue of TD-like targets
with compressed imaging, the sensitivity of the catalogue-mean endpoint

    g = d mean_j E[s_E,j | imaging_j, eta] / d eta = mean_j Cov_j(s_E, T),

where T are the eight frozen population scores and Cov_j is over target j's
selected, imaging-conditioned states. For each source class it computes the
exact expected Fisher matrix of the selected exponential-tilt mixture,

    F = E_data[(E[T | data] - E[T]) (E[T | data] - E[T])^T],

augmented by score terms for shared calibration offsets. Much of F at large
N comes from the shape of the state mixture, which the frozen base measure
dictates, so each case is also reported with a mean-only Fisher matrix,
J^T C^-1 J for the population means of the observables (J = Cov(x, T), C
their population-plus-noise covariance). Treat the mean-only numbers as the
conservative forecast and the mixture numbers as an optimistic bound. The
shared part of the target uncertainty after N sources is
sqrt(g^T (P0^-1 + N F)^-1 g); the per-lens state variance averages down with
catalogue size and is reported separately. A source-target contrast prior
adds its own floor.

Sources are spread evenly over cells: three foreground masses, with sizes
from a fixed size--mass relation, at five SN radii. Each source carries an
SN residual, a dynamical anchor, or both:

* SN: Gaussian response (default 0.15 mag, the TRAIN20 width near z=1.2),
  complete delivery, stellar microlensing variance, shared SN zero point;
* anchor (default "dispersion"): ln sigma_ap, the luminosity-weighted
  line-of-sight dispersion inside Re from the spherical Jeans equation
  (``ilsn.dynamics``), with a fractional error, a shared calibration offset,
  per-galaxy anisotropy scatter as extra width and the uncertain mean
  anisotropy added to the offset. "wolf" instead uses ln M_3D(<4/3 Re), the
  mass Wolf et al. (2010) show a global dispersion fixes nearly independently
  of anisotropy, with fractional mass errors. A broad finite core adds little
  mass inside Re, so either anchor reads the stars and halo, not the sheet.

The mean-only SN+anchor case uses only the population means of brightness and
anchor in each cell, which is what per-cell stacked spectra of the same
foregrounds measure; a low per-object signal-to-noise enters as a larger
anchor error.

Both use only states admitted for a single-image SN at that radius, so the
anchor-only control differs from the joint case only by the SN brightness;
that admission itself bounds the mass inside the SN radius. A second
control places the anchors on unselected galaxies of the same masses, so
that brightness and dynamics come from different objects.
Broad cores are normalized either by mass (package default: core mass
2 q_ref^2 eta M_ref at every q, so kappa_core at the ring falls as q^-2) or
by ring amplitude (q_ref=q, central surface density eta M_ref/(pi Re^2) at
every q). Delivery selection, dust and line-of-sight terms are omitted, so
this is a design comparison at one population point, not a survey forecast.

Targets have compressed imaging and, with ``--target-dispersion-error``, their
own aperture dispersion (TDCOSMO-like kinematics). At fixed imaging a
mass-sheet transformation scales H0 with s_E, so a shared SD divided by the
catalogue-mean s_E is the fractional H0 uncertainty it implies.

Run from the source directory after installing its dependencies::

    PYTHONPATH=. python examples/information_flow.py --output-dir outputs/information_flow
    PYTHONPATH=. python examples/information_flow.py --core-scales 5 10 20 --normalization ring
    PYTHONPATH=. python examples/information_flow.py --grid fine
    PYTHONPATH=. python examples/information_flow.py --target-dispersion-error 0.05

The 2,700-state standard grid with one core scale takes about two minutes on
four processes; each extra core scale, the 4,860-state primary grid or the
9,180-state fine grid adds proportionally. The closed forms reproduce
central finite differences of ``TargetSample.moments`` and of the mixture
log density.
"""

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from ilsn.dynamics import aperture_dispersion
from ilsn.lens_bank import build_lens_bank
from ilsn.lens_model import CompositeProfile, FiniteCore, GalaxyProfile
from ilsn.mass_grid import SCORE_NAMES, LensDesign, MassGrid, physical_state_parameters
from ilsn.observation import imaging_log_likelihood, td_selection
from ilsn.population import MassPopulation, TargetSample
from ilsn.sampling import gaussian_nodes

# Roman-quality target imaging resolves the core-amplitude axis: the target
# prior shared SD moves 0.022 -> 0.0275 -> 0.031 from 5 to 9 to 17 nodes and
# then stays put when halo nodes are refined, so use "fine" for headline numbers.
GRIDS = dict(
    standard=dict(eta_nodes=tuple(np.linspace(-0.2, 0.2, 5))),
    primary=dict(),
    fine=dict(eta_nodes=tuple(np.linspace(-0.2, 0.2, 17))),
)
PRIORS = dict(
    isotropic=(0.7,) * 8,
    # SHMR ~0.1 dex, concentration ~0.05 dex, IMF ~0.1 dex; broad sheet.
    physical=(0.5, 0.3, 0.4, 0.7, 0.7, 1.0, 0.7, 0.7),
)
TARGETS = dict(
    log_design_mass=np.log10([1.4e11, 1.8e11, 2.2e11, 2.6e11]),
    re_arcsec=(0.65, 0.8, 1.0, 1.2),
    z_lens=0.35,
    z_source=1.5,
)
SOURCE = dict(z_lens=0.3, z_source=1.2)
SIZE_RELATION = (11.3, 0.9, 0.6)  # pivot log10 mass, Re there (arcsec), slope
FIDUCIAL = dict(
    mass_multiplier=1.2,
    halo_residual=-0.5,
    concentration_residual=0.0,
    stellar_gradient=-0.15,
    halo_slope=1.0,
    eta_core=0.0,
)
SAMPLE_SIZES = (48, 192, 768, 3000)
FLOOR_N = 10**6  # stands in for an unlimited sample of that source class
ADMISSION = ("physical_valid", "single_image", "geometry_valid", "projection_valid")
# Dispersion anchor: luminosity-weighted sigma inside APERTURE_RE * Re for
# constant anisotropy, with a population mean, a per-galaxy scatter and an
# uncertain population mean (early-type galaxies: beta ~ 0-0.3).
APERTURE_RE = 1.0
ANISOTROPY = dict(mean=0.1, scatter=0.15, mean_sd=0.1)
# Fractional anchor errors and calibration offset: mass for "wolf", dispersion
# for "dispersion" (sigma errors are half the implied mass errors).
ANCHOR_DEFAULTS = dict(
    wolf=((0.1, 0.15, 0.3), 0.05), dispersion=((0.05, 0.1, 0.2), 0.025)
)


def core_grid(grid_name, q, normalization):
    """One-core-scale primary-axis grid with mass or ring normalization."""
    reference_q = q if normalization == "ring" else 5.0
    return MassGrid.primary(**GRIDS[grid_name], q_nodes=(q,), reference_q=reference_q)


def _state_profiles(design, grid):
    """Stars, halo and cores of every state of a one-object design."""
    p = physical_state_parameters(design, grid)
    for k in range(len(grid)):
        base = GalaxyProfile(
            design.z_lens[0],
            p["re_arcsec"][0],
            10 ** p["log_stellar_mass"][0, k],
            10 ** p["log_halo_mass"][0, k],
            p["concentration"][0, k],
            design.light_n[0],
            halo_inner_slope=p["halo_slope"][0, k],
            stellar_ml_gradient=p["stellar_gradient"][0, k],
        )
        cores = tuple(
            FiniteCore(r, s)
            for r, s in zip(
                p["core_radii_kpc"][0, k], p["sigma_core_components_msun_kpc2"][0, k]
            )
        )
        yield CompositeProfile(base, cores)


def wolf_mass(design, grid):
    """ln M_3D(<4/3 Re) in Msun for every state of a one-object design."""
    out = np.full(len(grid), np.nan)
    for k, profile in enumerate(_state_profiles(design, grid)):
        mass = float(profile.enclosed_mass_3d_msun(4 / 3 * profile.base.re_kpc))
        out[k] = np.log(mass) if mass > 0 else np.nan
    return out


def dispersion_anchor(design, grid, aperture_re):
    """ln sigma_ap (km/s) at the mean anisotropy and d ln sigma_ap / d beta.

    The aperture radius is aperture_re times the half-light radius.
    """
    scatter = ANISOTROPY["scatter"]
    beta = ANISOTROPY["mean"] + scatter * np.array([-1.0, 0.0, 1.0])
    value, slope = (np.full(len(grid), np.nan) for _ in range(2))
    for k, profile in enumerate(_state_profiles(design, grid)):
        aperture = aperture_re * profile.base.re_kpc
        ln_sigma = np.log(aperture_dispersion(profile, aperture, beta))
        if np.all(np.isfinite(ln_sigma)):
            value[k] = ln_sigma[1]
            slope[k] = (ln_sigma[2] - ln_sigma[0]) / (2 * scatter)
    return value, slope


def source_size(log_mass):
    """Assigned half-light radius in arcsec from a fixed size--mass relation."""
    pivot, re, slope = SIZE_RELATION
    return re * 10 ** (slope * (log_mass - pivot))


def _bank_job(job):
    grid_name, q, normalization, kind, design_kwargs, anchor, aperture_re, target_re = (
        job
    )
    grid = core_grid(grid_name, q, normalization)
    design = LensDesign(**design_kwargs)
    if kind == "target":
        bank = build_lens_bank(design, grid, source=False)
        if target_re is not None:
            value, slope = dispersion_anchor(design, grid, target_re)
            bank.update(ln_dispersion=value[None, :], dispersion_slope=slope[None, :])
        return bank
    if kind == "anchor":
        if anchor == "wolf":
            value, slope = wolf_mass(design, grid), np.zeros(len(grid))
        else:
            value, slope = dispersion_anchor(design, grid, aperture_re)
        return dict(ln_anchor=value[None, :], anchor_slope=slope[None, :])
    return build_lens_bank(design, grid, target=False)


def build_banks(
    grid_name,
    core_scales,
    normalization,
    radii,
    masses,
    processes=4,
    anchor="dispersion",
    aperture_re=None,
    target_aperture_re=None,
):
    """Target, source-cell and anchor arrays concatenated over core-scale blocks.

    Source cells are every foreground mass (with its assigned size) at every
    SN radius; anchors depend on the foreground mass only. For the dispersion
    anchor, per-galaxy anisotropy scatter becomes an extra per-state width and
    the uncertain population-mean anisotropy a shared offset (its slope
    averaged over states with a finite anchor). The aperture defaults to
    APERTURE_RE half-light radii. With target_aperture_re, target banks also
    hold their own aperture dispersions for TDCOSMO-like target kinematics.
    """
    aperture_re = APERTURE_RE if aperture_re is None else aperture_re
    designs = [
        (
            "target",
            dict(
                log_design_mass=m,
                re_arcsec=r,
                z_lens=TARGETS["z_lens"],
                z_source=TARGETS["z_source"],
            ),
        )
        for m, r in zip(TARGETS["log_design_mass"], TARGETS["re_arcsec"])
    ]
    cells = [(m, r) for m in masses for r in radii]
    foreground = {
        m: dict(SOURCE, log_design_mass=m, re_arcsec=source_size(m)) for m in masses
    }
    designs += [("source", dict(foreground[m], radius_arcsec=r)) for m, r in cells]
    designs += [("anchor", foreground[m]) for m in masses]
    jobs = [
        (
            grid_name,
            q,
            normalization,
            kind,
            kwargs,
            anchor,
            aperture_re,
            target_aperture_re,
        )
        for q in core_scales
        for kind, kwargs in designs
    ]
    with Pool(processes) as pool:
        results = pool.map(_bank_job, jobs)
    banks = []
    for i in range(len(designs)):
        blocks = results[i :: len(designs)]
        banks.append(
            {key: np.concatenate([b[key] for b in blocks], axis=1) for key in blocks[0]}
        )
    grids = [core_grid(grid_name, q, normalization) for q in core_scales]
    base = np.concatenate([g.base_probability / len(grids) for g in grids])
    scores = np.vstack([g.scores for g in grids])
    fiducial = np.zeros(len(base), bool)
    fiducial[: len(grids[0])] = np.logical_and.reduce(
        [np.isclose(getattr(grids[0], k), v) for k, v in FIDUCIAL.items()]
    )
    if fiducial.sum() != 1:
        raise ValueError("Grid must contain exactly one fiducial state")
    n, c = len(TARGETS["re_arcsec"]), len(cells)
    anchors = dict(zip(masses, banks[n + c :]))
    slopes = []
    for b in anchors.values():
        finite = np.isfinite(b["ln_anchor"][0])
        slopes.append(base[finite] @ b["anchor_slope"][0, finite] / base[finite].sum())
    return dict(
        population=MassPopulation(base, scores),
        fiducial=int(np.flatnonzero(fiducial)[0]),
        targets=banks[:n],
        cells=[
            dict(
                log_mass=m,
                radius=r,
                bank=b,
                anchor=anchors[m]["ln_anchor"][0],
                anchor_width=abs(anchors[m]["anchor_slope"][0]) * ANISOTROPY["scatter"],
            )
            for (m, r), b in zip(cells, banks[n : n + c])
        ],
        anchor=anchor,
        anchor_offset_sd=(
            float(abs(np.mean(slopes)) * ANISOTROPY["mean_sd"])
            if anchor == "dispersion"
            else 0.0
        ),
    )


def target_sensitivity(
    population, banks, fiducial, theta_error, xi_error, eta0, dispersion_error=None
):
    """Per-target E[s_E], its state SD and gradient g_j = Cov_j(s_E, T).

    With dispersion_error, each target also has its own aperture dispersion,
    observed at the fiducial state with that fractional error plus the
    per-galaxy anisotropy scatter. Its calibration offset is the one the
    source anchors share, so E[s_E] also responds to it:
    offset_gradient = Cov_j(s_E, d ln L_j / d offset).
    """
    logw = population.log_probabilities(eta0)[0]
    rows = []
    for bank in banks:
        valid = bank["physical_valid"] & bank["geometry_valid"] & bank["endpoint_valid"]
        theta, xi = bank["theta_e_arcsec"], bank["xi_E"]
        selection = td_selection(
            theta,
            bank["caustic_area_arcsec2"],
            np.array([TARGETS["z_lens"]]),
            np.array([TARGETS["z_source"]]),
            valid,
        )
        ot, ox = theta[0, fiducial], xi[0, fiducial]
        logl = imaging_log_likelihood(
            theta, xi, [ot], [ox], [theta_error * ot], [xi_error]
        )
        offset_score = np.zeros(theta.shape[1])
        if dispersion_error is not None:
            ln_sigma = bank["ln_dispersion"]
            valid = valid & np.isfinite(ln_sigma)
            width = np.hypot(
                dispersion_error,
                np.nan_to_num(bank["dispersion_slope"]) * ANISOTROPY["scatter"],
            )
            residual = np.nan_to_num(ln_sigma) - ln_sigma[0, fiducial]
            logl = logl - 0.5 * (residual / width) ** 2 - np.log(width)
            offset_score = -(residual / width**2)[0]
        sample = TargetSample(bank["s_E"], selection, logl, valid)
        with np.errstate(divide="ignore"):
            joint = logw + np.log(sample.selection[0]) + sample.log_likelihood[0]
        post = np.exp(joint - logsumexp(joint))
        s = sample.endpoint[0]
        mean = post @ s
        check = sample.moments(population, eta0[None, :])["mean"][0, 0]
        if not np.isclose(mean, check, rtol=0, atol=1e-10):
            raise RuntimeError("Closed-form target mean disagrees with TargetSample")
        centered = population.scores - post @ population.scores
        rows.append(
            dict(
                theta_e=float(ot),
                xi_e=float(ox),
                mean=float(mean),
                state_sd=float(np.sqrt(post @ (s - mean) ** 2)),
                gradient=(post * (s - mean)) @ centered,
                offset_gradient=float(post @ ((s - mean) * offset_score)),
                sample=sample,
            )
        )
    return rows


def prior_spread(population, targets, prior_sd, n=1024, seed=0):
    """Nonlinear prior SD of the catalogue-mean endpoint over Gaussian tilts."""
    nodes, weights = gaussian_nodes(
        np.zeros(len(prior_sd)), np.diag(prior_sd**2), n, seed
    )
    mean = np.mean(
        [t["sample"].moments(population, nodes)["mean"][:, 0] for t in targets], axis=0
    )
    center = weights @ mean
    return float(np.sqrt(weights @ (mean - center) ** 2))


def _normal_log(x, center, width):
    return -0.5 * ((x[:, None] - center) / width) ** 2 - np.log(
        np.sqrt(2 * np.pi) * width
    )


def _grid_for(center, width, per_width=8, pad=6.0):
    step = width.min() / per_width
    low, high = center.min() - pad * width.max(), center.max() + pad * width.max()
    n = int(np.ceil((high - low) / step)) + 1
    return np.linspace(low, high, n)


def source_fisher(logw, scores, sn=None, anchor=None):
    """Expected Fisher of (eta, SN zero point, anchor offset) for one source.

    :param logw: Log state probabilities of admitted states at eta0.
    :param scores: Population scores of admitted states, shape (state, 8).
    :param sn: Optional (shift_mag, width_mag) arrays over admitted states.
    :param anchor: Optional (value, width) arrays over admitted states.
    :return: 10 by 10 Fisher matrix and the quadrature mass (should be 1).
    """
    logw = logw - logsumexp(logw)
    w = np.exp(logw)
    centered = scores - w @ scores
    dim = scores.shape[1]
    F = np.zeros((dim + 2, dim + 2))
    observed = [(i, o) for i, o in enumerate((sn, anchor)) if o is not None]
    if not observed:
        raise ValueError("At least one observable required")
    grids = [_grid_for(*o) for _, o in observed]
    kernels = [_normal_log(x, *o) for x, (_, o) in zip(grids, observed)]
    steps = [x[1] - x[0] for x in grids]

    def accumulate(joint, residuals, weight):
        lognorm = logsumexp(joint, axis=1)
        r = np.exp(joint - lognorm[:, None])
        S = np.zeros((len(joint), dim + 2))
        S[:, :dim] = r @ centered
        for (slot, (_, width)), res in zip(observed, residuals):
            S[:, dim + slot] = np.sum(r * res / width**2, axis=1)
        p = np.exp(lognorm) * weight
        return (S.T * p) @ S, p.sum()

    first_residual = grids[0][:, None] - observed[0][1][0]
    mass = 0.0
    if len(observed) == 1:
        F, mass = accumulate(logw + kernels[0], [first_residual], steps[0])
    else:
        for x_b, k_b in zip(grids[1], kernels[1]):
            joint = logw + kernels[0] + k_b
            res_b = np.broadcast_to(x_b - observed[1][1][0], first_residual.shape)
            dF, dm = accumulate(joint, [first_residual, res_b], steps[0] * steps[1])
            F += dF
            mass += dm
    return F, mass


def moment_fisher(logw, scores, sn=None, anchor=None):
    """Fisher of the sample means alone, as for a Gaussian of fixed covariance.

    This keeps only how eta moves the mean observables, d E[x]/d eta =
    Cov(x, T), against their population-plus-noise covariance. Unlike
    :func:`source_fisher` it ignores information in the shape of the state
    mixture, which depends on the frozen base measure and grows without limit
    for large samples.
    """
    w = np.exp(logw - logsumexp(logw))
    centered = scores - w @ scores
    dim = scores.shape[1]
    observed = [(i, o) for i, o in enumerate((sn, anchor)) if o is not None]
    values = np.column_stack([o[0] for _, o in observed])
    values = values - w @ values
    covariance = (values.T * w) @ values + np.diag([w @ o[1] ** 2 for _, o in observed])
    jacobian = np.zeros((len(observed), dim + 2))
    jacobian[:, :dim] = (values.T * w) @ centered
    for row, (slot, _) in enumerate(observed):
        jacobian[row, dim + slot] = 1.0
    return jacobian.T @ np.linalg.solve(covariance, jacobian)


def shared_sd(g, fisher, prior_precision, contrast_sd=0.0):
    """Linearized shared SD of the catalogue-mean endpoint."""
    covariance = np.linalg.inv(prior_precision + fisher)
    ge = np.concatenate((g, np.zeros(len(covariance) - len(g))))
    contrast = contrast_sd * np.linalg.norm(ge[: len(SCORE_NAMES)])
    return float(np.sqrt(ge @ covariance @ ge + contrast**2))


def analyse(
    banks,
    radii,
    *,
    theta_error,
    xi_error,
    sn_sigma,
    micro_scale,
    anchor_errors,
    prior,
    sn_zero_point,
    anchor_zero_point,
    contrast_sd,
    target_dispersion_error=None,
):
    population = banks["population"]
    eta0 = np.zeros(len(SCORE_NAMES))
    targets = target_sensitivity(
        population,
        banks["targets"],
        banks["fiducial"],
        theta_error,
        xi_error,
        eta0,
        target_dispersion_error,
    )
    g = np.mean([t["gradient"] for t in targets], axis=0)
    p0 = np.asarray(PRIORS[prior], float)
    anchor_offset = np.hypot(anchor_zero_point, banks["anchor_offset_sd"])
    precision = np.diag(
        np.concatenate((p0**-2, [sn_zero_point**-2, anchor_offset**-2]))
    )
    # Targets with their own dispersion also respond to the shared offset.
    g_offset = float(np.mean([t["offset_gradient"] for t in targets]))
    g_full = np.concatenate((g, [0.0, g_offset]))
    contribution = np.append((g * p0) ** 2, (g_offset * anchor_offset) ** 2)
    budget = dict(
        zip(SCORE_NAMES + ("dispersion_offset",), contribution / contribution.sum())
    )
    prior_shared = float(np.sqrt(contribution.sum()))
    individual = float(np.sqrt(np.mean([t["state_sd"] ** 2 for t in targets])))

    logw_all = population.log_probabilities(eta0)[0]
    fishers, per_source = {}, []
    for index, cell in enumerate(banks["cells"]):
        bank, radius = cell["bank"], cell["radius"]
        admitted = np.logical_and.reduce([bank[k][0] for k in ADMISSION])
        admitted &= np.isfinite(cell["anchor"])
        kappa, mean = bank["kappa"][0, admitted], bank["mean_kappa"][0, admitted]
        shift = 2.5 * np.log10((1 - mean) * (1 - 2 * kappa + mean))
        width = np.sqrt(sn_sigma**2 + micro_scale**2 * bank["kappa_star"][0, admitted])
        logw, scores = logw_all[admitted], population.scores[admitted]
        wa = np.exp(logw - logsumexp(logw))
        anchor, spread = cell["anchor"][admitted], cell["anchor_width"][admitted]
        sn = (shift, width)
        fishers["mixture", "SN", index, None], m1 = source_fisher(logw, scores, sn=sn)
        fishers["mean", "SN", index, None] = moment_fisher(logw, scores, sn=sn)
        mass = [m1]
        for err in anchor_errors:
            a = (anchor, np.hypot(err, spread))
            for kind, data in (
                ("anchor", dict(anchor=a)),
                ("SN+anchor", dict(sn=sn, anchor=a)),
            ):
                fishers["mixture", kind, index, err], m = source_fisher(
                    logw, scores, **data
                )
                fishers["mean", kind, index, err] = moment_fisher(logw, scores, **data)
                mass.append(m)
        F = fishers["mixture", "SN", index, None][:8, :8]
        eig, vec = np.linalg.eigh(F)
        top = vec[:, -1] * np.sign(vec[np.argmax(abs(vec[:, -1])), -1])
        per_source.append(
            dict(
                log_design_mass=cell["log_mass"],
                radius_arcsec=radius,
                admitted_fraction=float(population.base_probability @ admitted),
                mean_shift_mag=float(wa @ shift),
                sd_shift_mag=float(np.sqrt(wa @ (shift - wa @ shift) ** 2)),
                sd_ln_anchor=float(np.sqrt(wa @ (anchor - wa @ anchor) ** 2)),
                sn_fisher_eigenvalues=eig[::-1].tolist(),
                sn_top_direction=dict(zip(SCORE_NAMES, top.tolist())),
                quadrature_mass_range=[float(min(mass)), float(max(mass))],
            )
        )
        # Control: as many anchors on unselected galaxies of this mass, so that
        # dynamics and SN brightness come from different objects.
        if ("mean", "survey", cell["log_mass"], anchor_errors[0]) not in fishers:
            physical = bank["physical_valid"][0] & np.isfinite(cell["anchor"])
            a, a_spread = cell["anchor"][physical], cell["anchor_width"][physical]
            data = (logw_all[physical], population.scores[physical])
            for err in anchor_errors:
                a_err = (a, np.hypot(err, a_spread))
                key = ("survey", cell["log_mass"], err)
                fishers[("mixture",) + key] = source_fisher(*data, anchor=a_err)[0]
                fishers[("mean",) + key] = moment_fisher(*data, anchor=a_err)

    cells = banks["cells"]

    def combined(info, kind, err, mix):
        if kind == "SN, survey":
            survey = sum(
                f * fishers[info, "survey", cells[i]["log_mass"], err]
                for i, f in mix.items()
            )
            return combined(info, "SN", None, mix) + survey
        return sum(f * fishers[info, kind, i, err] for i, f in mix.items())

    designs = {}
    for r in radii:
        chosen = [i for i, c in enumerate(cells) if c["radius"] == r]
        designs[f'R={r:g}"'] = {i: 1 / len(chosen) for i in chosen}
    designs["all cells"] = {i: 1 / len(cells) for i in range(len(cells))}
    rows = []
    for label, mix in designs.items():
        cases = [("SN", None)] + [
            (kind, err)
            for err in anchor_errors
            for kind in ("anchor", "SN, survey", "SN+anchor")
        ]
        for info in ("mean", "mixture"):
            for kind, err in cases:
                Fm = combined(info, kind, err, mix)
                for n in SAMPLE_SIZES + (FLOOR_N,):
                    rows.append(
                        dict(
                            information=info,
                            design=label,
                            data=kind,
                            anchor_error=err,
                            n_sources=n,
                            shared_sd=shared_sd(g_full, n * Fm, precision),
                            shared_sd_with_contrast=shared_sd(
                                g_full, n * Fm, precision, contrast_sd
                            ),
                            eta_core_mean_sd=float(
                                np.sqrt(np.linalg.inv(precision + n * Fm)[5, 5])
                            ),
                        )
                    )
    return dict(
        targets=[
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in t.items()
                if k != "sample"
            }
            for t in targets
        ],
        gradient=dict(zip(SCORE_NAMES, g.tolist())),
        offset_gradient=g_offset,
        budget=budget,
        prior_shared_sd=prior_shared,
        prior_shared_sd_nonlinear=prior_spread(population, targets, p0),
        prior_shared_sd_with_contrast=shared_sd(
            g_full, np.zeros_like(precision), precision, contrast_sd
        ),
        anchor_offset_sd=float(anchor_offset),
        catalogue_mean_s_E=float(np.mean([t["mean"] for t in targets])),
        individual_state_sd=individual,
        sources=per_source,
        rows=rows,
    )


def plot(result, output_dir, title):
    rows = result["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    names = list(result["budget"])
    axes[0].barh(range(len(names)), [result["budget"][n] for n in names], color="0.35")
    axes[0].set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("share of prior shared variance")
    axes[0].set_title(f"prior shared SD {result['prior_shared_sd']:.4f}")
    errors = sorted({r["anchor_error"] for r in rows if r["anchor_error"] is not None})
    err = errors[len(errors) // 2] if errors else None
    styles = [
        ("SN", None, "-", "#2a78d6"),
        ("anchor", err, "--", "#eb6834"),
        ("SN+anchor", err, "-.", "#1baf7a"),
    ]
    for kind, e, ls, color in styles:
        sel = [
            r
            for r in rows
            if r["design"] == "all cells"
            and r["data"] == kind
            and r["anchor_error"] == e
            and r["n_sources"] != FLOOR_N
            and r["information"] == "mean"
        ]
        if sel:
            label = kind if e is None else f"{kind} ({e:.0%})"
            axes[1].plot(
                [r["n_sources"] for r in sel],
                [r["shared_sd"] for r in sel],
                ls,
                color=color,
                marker="o",
                label=label,
            )
    axes[1].axhline(
        result["prior_shared_sd"], color="0.5", lw=0.8, ls=":", label="prior"
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of sources (all cells)")
    axes[1].set_ylabel("shared SD of catalogue-mean $s_E$ (mean-only)")
    axes[1].legend(frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"information_flow.{suffix}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/information_flow")
    )
    parser.add_argument("--grid", choices=sorted(GRIDS), default="standard")
    parser.add_argument("--core-scales", type=float, nargs="+", default=[5.0])
    parser.add_argument("--normalization", choices=("mass", "ring"), default="mass")
    parser.add_argument(
        "--radii", type=float, nargs="+", default=[2.5, 3.5, 5.0, 7.0, 10.0]
    )
    parser.add_argument(
        "--source-masses", type=float, nargs="+", default=[11.0, 11.3, 11.6]
    )
    parser.add_argument("--theta-error", type=float, default=0.02)
    parser.add_argument("--xi-error", type=float, default=0.05)
    parser.add_argument("--sn-sigma", type=float, default=0.15)
    parser.add_argument("--micro-scale", type=float, default=0.2)
    parser.add_argument(
        "--anchor", choices=sorted(ANCHOR_DEFAULTS), default="dispersion"
    )
    parser.add_argument("--anchor-errors", type=float, nargs="*", default=None)
    parser.add_argument("--aperture-re", type=float, default=APERTURE_RE)
    parser.add_argument("--target-dispersion-error", type=float, default=None)
    parser.add_argument("--target-aperture-re", type=float, default=1.0)
    parser.add_argument("--prior", choices=sorted(PRIORS), default="isotropic")
    parser.add_argument("--sn-zero-point", type=float, default=0.01)
    parser.add_argument("--anchor-zero-point", type=float, default=None)
    parser.add_argument("--contrast-sd", type=float, default=0.1)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    errors, offset = ANCHOR_DEFAULTS[args.anchor]
    if args.anchor_errors is None:
        args.anchor_errors = list(errors)
    if args.anchor_zero_point is None:
        args.anchor_zero_point = offset
    banks = build_banks(
        args.grid,
        args.core_scales,
        args.normalization,
        args.radii,
        args.source_masses,
        args.processes,
        args.anchor,
        args.aperture_re,
        None if args.target_dispersion_error is None else args.target_aperture_re,
    )
    result = analyse(
        banks,
        args.radii,
        theta_error=args.theta_error,
        xi_error=args.xi_error,
        sn_sigma=args.sn_sigma,
        micro_scale=args.micro_scale,
        anchor_errors=args.anchor_errors,
        prior=args.prior,
        sn_zero_point=args.sn_zero_point,
        anchor_zero_point=args.anchor_zero_point,
        contrast_sd=args.contrast_sd,
        target_dispersion_error=args.target_dispersion_error,
    )
    result["settings"] = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    result["states"] = len(banks["population"].base_probability)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "information_flow.json").write_text(json.dumps(result, indent=1))
    with open(args.output_dir / "information_flow.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    title = (
        f"{args.grid} grid, q={args.core_scales} ({args.normalization}), "
        f"imaging {args.theta_error:.0%}/{args.xi_error}, {args.prior} prior, "
        f"{args.anchor} anchor"
        + (
            ""
            if args.target_dispersion_error is None
            else f", target sigma {args.target_dispersion_error:.0%}"
        )
    )
    plot(result, args.output_dir, title)

    print(f"{result['states']} states; {title}")
    print(
        f"prior shared SD {result['prior_shared_sd']:.4f} (linear), "
        f"{result['prior_shared_sd_nonlinear']:.4f} (over prior tilts); "
        f"mean per-lens state SD "
        f"{result['individual_state_sd']:.4f}; anchor offset SD "
        f"{result['anchor_offset_sd']:.4f}"
    )
    print(
        f"catalogue-mean s_E {result['catalogue_mean_s_E']:.3f}; at fixed imaging a "
        "mass-sheet transformation scales H0 with s_E, so SD / mean s_E is the "
        "fractional H0 uncertainty"
    )
    for name in SCORE_NAMES:
        print(
            f"  {name:28s} g={result['gradient'][name]:+.4f} "
            f"share={result['budget'][name]:.3f}"
        )
    print(
        f"  {'dispersion_offset':28s} g={result['offset_gradient']:+.4f} "
        f"share={result['budget']['dispersion_offset']:.3f}"
    )
    for s in result["sources"]:
        print(
            f"  logM={s['log_design_mass']:g} R={s['radius_arcsec']:g}\": admitted {s['admitted_fraction']:.2f}, "
            f"SN Fisher eig {np.round(s['sn_fisher_eigenvalues'][:3], 3)}, "
            f"quadrature mass {np.round(s['quadrature_mass_range'], 4)}"
        )
    print(
        f"{'design':>10s} {'data':>16s} "
        + " ".join(f"N={n:<6d}" for n in SAMPLE_SIZES)
        + " N=inf"
    )
    print(
        "posterior SD of the eta_core_mean tilt (prior "
        f"{PRIORS[args.prior][5]}), all cells:"
    )
    for info in ("mean", "mixture"):
        for key in dict.fromkeys(
            (r["data"], r["anchor_error"]) for r in result["rows"]
        ):
            sel = [
                r
                for r in result["rows"]
                if r["design"] == "all cells"
                and (r["data"], r["anchor_error"]) == key
                and r["information"] == info
            ]
            tag = key[0] if key[1] is None else f"{key[0]} {key[1]:.0%}"
            print(
                f"{info:>10s} {tag:>16s} "
                + " ".join(f"{r['eta_core_mean_sd']:.3f}  " for r in sel)
            )
    for info in ("mean", "mixture"):
        print(f"[{info} information]")
        for label in dict.fromkeys(r["design"] for r in result["rows"]):
            for key in dict.fromkeys(
                (r["data"], r["anchor_error"]) for r in result["rows"]
            ):
                sel = [
                    r
                    for r in result["rows"]
                    if r["design"] == label
                    and (r["data"], r["anchor_error"]) == key
                    and r["information"] == info
                ]
                tag = key[0] if key[1] is None else f"{key[0]} {key[1]:.0%}"
                print(
                    f"{label:>10s} {tag:>16s} "
                    + " ".join(f"{r['shared_sd']:.4f} " for r in sel)
                )


if __name__ == "__main__":
    main()
