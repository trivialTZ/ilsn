"""Plot one simulated Roman training SN's conditional brightness likelihood.

Run from the public repository root::

    PYTHONPATH=. python examples/roman_magnification.py --output-dir outputs/roman_magnification

This event belongs to the calibration training simulation. Its interval is
defined by a log-likelihood drop, rather than an asserted coverage probability.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ilsn.likelihood import magnification_log_likelihood
from ilsn.response import ResponseCalibration


def likelihood_interval(magnification, log_likelihood):
    """Locate the likelihood peak and its half-unit log-drop interval.

    A local parabola refines the peak on the plotted grid; linear interpolation
    locates both threshold crossings. These are likelihood summaries without
    a mass prior or a magnification-density Jacobian.

    :param magnification: Increasing positive magnification grid.
    :param log_likelihood: Conditional log likelihood on that grid.
    :return: Peak magnification, peak log likelihood, lower and upper crossings.
    :rtype: tuple
    """
    index = int(np.argmax(log_likelihood))
    if not 0 < index < len(magnification) - 1:
        raise ValueError("The magnification grid does not bracket the likelihood peak")
    origin = magnification[index]
    curve = np.polyfit(
        magnification[index - 1 : index + 2] - origin,
        log_likelihood[index - 1 : index + 2],
        2,
    )
    peak = origin - curve[1] / (2 * curve[0])
    peak_log_likelihood = float(np.polyval(curve, peak - origin))
    threshold = peak_log_likelihood - 0.5
    if log_likelihood[0] >= threshold or log_likelihood[-1] >= threshold:
        raise ValueError(
            "The magnification grid does not bracket the log-drop interval"
        )
    lower = np.interp(
        threshold, log_likelihood[: index + 1], magnification[: index + 1]
    )
    upper = np.interp(
        threshold, log_likelihood[index:][::-1], magnification[index:][::-1]
    )
    return peak, peak_log_likelihood, lower, upper


def main():
    """Save the Roman likelihood figure and report its numerical readout."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/roman_magnification"),
        help="Directory for the PNG and PDF figures",
    )
    args = parser.parse_args()

    calibration_path = (
        Path(__file__).resolve().parents[1] / "data" / "roman_response.npz"
    )
    calibration = ResponseCalibration.load(calibration_path)
    residual_mag, true_redshift = -0.3513132671710686, 1.2
    injected_magnification = 10**0.08
    magnification = np.linspace(0.67, 2.15, 301)
    log_likelihood = magnification_log_likelihood(
        residual_mag,
        true_redshift,
        magnification,
        calibration,
    )
    peak, peak_log_likelihood, lower, upper = likelihood_interval(
        magnification, log_likelihood
    )
    relative_likelihood = np.exp(log_likelihood - peak_log_likelihood)
    fractional_half_width = (upper - lower) / (2 * peak)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.frameon": False,
        }
    )
    blue, orange = "#2b5b8e", "#c77932"
    figure, axis = plt.subplots(figsize=(6.5, 4.6))
    axis.plot(
        magnification,
        relative_likelihood,
        color=blue,
        linewidth=2.2,
        label="Brightness likelihood",
    )
    interval_grid = np.r_[
        lower, magnification[(magnification > lower) & (magnification < upper)], upper
    ]
    axis.fill_between(
        interval_grid,
        0.0,
        np.interp(interval_grid, magnification, relative_likelihood),
        color=blue,
        alpha=0.14,
        label=r"$\Delta\ln L\leq 0.5$",
    )
    axis.axvline(
        injected_magnification,
        color=orange,
        linewidth=1.8,
        linestyle="--",
        label=rf"Injected $\mu={injected_magnification:.2f}$",
    )
    axis.axvline(peak, color=blue, linewidth=1.0, linestyle=":", alpha=0.7)
    axis.set(
        xlabel=r"Flux magnification $\mu$",
        ylabel="Likelihood / peak likelihood",
        xlim=(magnification[0], magnification[-1]),
        ylim=(0.0, 1.12),
    )
    axis.set_title("Roman SN magnification", loc="left", pad=12)
    axis.text(
        0.97,
        0.96,
        rf"Peak $\mu={peak:.2f}$"
        + "\n"
        + rf"Fractional half-width ${fractional_half_width:.3f}$",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    axis.legend(loc="upper left", fontsize=9.5)
    figure.text(
        0.12,
        0.035,
        "One simulated training SN at true redshift 1.2; no coverage claim.",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.065, 1.0, 1.0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"roman_magnification.{extension}", dpi=220)
    plt.close(figure)
    print(
        f"Likelihood peak {peak:.4f}; injected magnification {injected_magnification:.4f}; "
        f"fractional half-width {fractional_half_width:.4f}."
    )


if __name__ == "__main__":
    main()
