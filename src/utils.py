import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks, periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

COLOR_PRIMARY = "steelblue"
COLOR_SECONDARY = "darkorange"
COLOR_TERTIARY = "cadetblue"
COLOR_ACCENT = "indianred"
COLOR_REFERENCE = "dimgray"
COLOR_MEAN = "black"

PLOT_TITLE_SIZE = 12
PLOT_LABEL_SIZE = 11


def check_convergence(df):
    is_na = df["energy"].isna()

    na_groups = (~is_na).cumsum()[is_na]
    lengths_consecutive_na = na_groups.groupby(na_groups).size()
    print("\n== Series Convergence Check ==")
    print(lengths_consecutive_na.value_counts().sort_index())


def plot_periodogram_log(periods, psd, peak_idx):
    seasonal_markers = [
        (24, "1 day"),
        (24 * 7, "1 week"),
        (24 * 31, "1 month"),
        (24 * 31 * 6, "6 months"),
        (24 * 365, "1 year"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax in axes:
        ax.plot(periods, psd, lw=2.0, color=COLOR_PRIMARY, label="Periodogram")
        ax.scatter(
            periods[peak_idx],
            psd[peak_idx],
            color=COLOR_ACCENT,
            marker="X",
            s=80,
            zorder=3,
            label="Peaks",
        )
        for period_h, label in seasonal_markers:
            ax.axvline(
                period_h, color=COLOR_REFERENCE, ls="--", alpha=0.7, label="label"
            )
        ax.set_ylabel("Power", fontsize=PLOT_LABEL_SIZE)
        ax.set_xlabel("Period (hours)", fontsize=PLOT_LABEL_SIZE)
        ax.grid(True, alpha=0.3)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[1].set_xscale("log")
    fig.suptitle("Energy Consumption Periodogram", fontsize=14)
    axes[0].set_title("Log-log scale", fontsize=PLOT_TITLE_SIZE)
    axes[1].set_title("Semi-log scale", fontsize=PLOT_TITLE_SIZE)

    plt.tight_layout()
    plt.show()


def plot_periodogram_normal(
    periods,
    psd,
    peak_idx,
    short_idx=[0, 100],
    long_idx=None,
    fig_title="Energy Consumption Periodogram – Detailed View",
    short_title=None,
    long_title=None,
    ylabel="Power",
):

    if long_idx is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        axes = [ax]
        masks_and_peaks = [
            (
                (periods >= short_idx[0]) & (periods <= short_idx[1]),
                peak_idx[
                    (periods[peak_idx] >= short_idx[0])
                    & (periods[peak_idx] <= short_idx[1])
                ],
                short_title or f"Periods {short_idx[0]}–{short_idx[1]} hours",
            )
        ]

    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        masks_and_peaks = [
            (
                (periods >= short_idx[0]) & (periods <= short_idx[1]),
                peak_idx[
                    (periods[peak_idx] >= short_idx[0])
                    & (periods[peak_idx] <= short_idx[1])
                ],
                short_title or f"Short-term periods ({short_idx[0]} – {short_idx[1]})",
            ),
            (
                (periods >= long_idx[0]) & (periods <= long_idx[1]),
                peak_idx[
                    (periods[peak_idx] >= long_idx[0])
                    & (periods[peak_idx] <= long_idx[1])
                ],
                long_title or f"Long-term periods ({long_idx[0]} – {long_idx[1]})",
            ),
        ]

    for ax, (mask, peak_subset, title) in zip(axes, masks_and_peaks):
        ax.plot(
            periods[mask], psd[mask], lw=2.0, color=COLOR_PRIMARY, label="Periodogram"
        )
        ax.scatter(
            periods[peak_subset],
            psd[peak_subset],
            marker="X",
            s=80,
            color=COLOR_ACCENT,
            zorder=3,
            label="Peaks",
        )
        xmin, xmax = periods[mask].min(), periods[mask].max()
        ax.set_xlim(xmin, xmax)
        default_ticks = [t for t in ax.get_xticks() if xmin <= t <= xmax]
        peak_ticks = [round(x, 0) for x in periods[peak_subset]]
        ax.set_xticks(default_ticks + peak_ticks)
        ax.vlines(
            x=periods[peak_subset],
            ymin=0,
            ymax=psd[peak_subset],
            linestyle="--",
            color=COLOR_ACCENT,
            alpha=0.7,
        )
        ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_SIZE)
        ax.set_xlabel("Period (hours)", fontsize=PLOT_LABEL_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        ax.set_title(title, fontsize=PLOT_TITLE_SIZE)

    fig.suptitle(fig_title, fontsize=14)
    plt.tight_layout()
    plt.show()


def compute_periodogram(series, prominence_frac=0.25):
    freqs, psd = periodogram(series, fs=1)
    periods = 1 / freqs[1:]
    psd = psd[1:]
    peak_idx, _ = find_peaks(psd, prominence=np.max(psd) * prominence_frac)
    peak_idx = peak_idx.astype(int)
    return periods, psd, peak_idx


def plot_mstl_decomposition(decomp, periods, title=None):
    fig = decomp.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle(
        title or f"MSTL Decomposition – periods {list(periods)}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def compute_seasonal_strength(decomp):
    seasonal_components = decomp.seasonal
    residuals = decomp.resid
    var_residuals = np.var(residuals, ddof=1)
    strengths = {}
    for season in seasonal_components:
        var_seasonal_plus_residuals = np.var(
            seasonal_components[season] + residuals, ddof=1
        )
        strengths[season] = max(0, 1 - var_residuals / var_seasonal_plus_residuals)
    return strengths


def plot_seasonal_correlation(decomp, title="Correlation Between Seasonal Components"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        decomp.seasonal.corr(),
        cmap="mako",
        annot=True,
        fmt=".2f",
    )
    plt.title(title, fontsize=PLOT_TITLE_SIZE)
    plt.tight_layout()
    plt.show()


def plot_acf_pacf_residuals(
    series,
    lags_short=100,
    lags_seasonal=None,
    series_label="residuals",
    fig_title=None,
):
    if lags_seasonal is None:
        lags_seasonal = np.arange(0, 512, 24)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    plot_acf(series, lags=lags_short, alpha=0.05, ax=axes[0, 0])
    axes[0, 0].set_title(
        f"ACF – {series_label} (lags 0–{lags_short})", fontsize=PLOT_TITLE_SIZE
    )

    plot_pacf(series, lags=lags_short, alpha=0.05, ax=axes[1, 0])
    axes[1, 0].set_title(
        f"PACF – {series_label} (lags 0–{lags_short})", fontsize=PLOT_TITLE_SIZE
    )

    plot_acf(series, lags=lags_seasonal, alpha=0.05, ax=axes[0, 1])
    axes[0, 1].set_title(
        f"ACF – {series_label} (lags 0–{lags_seasonal[-1]})", fontsize=PLOT_TITLE_SIZE
    )

    plot_pacf(series, lags=lags_seasonal, alpha=0.05, ax=axes[1, 1])
    axes[1, 1].set_title(
        f"PACF – {series_label} (lags 0–{lags_seasonal[-1]})", fontsize=PLOT_TITLE_SIZE
    )

    fig.suptitle(fig_title or f"ACF/PACF of {series_label.title()}", fontsize=14)
    plt.tight_layout()
    plt.show()
