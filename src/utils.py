"""
utils.py
--------
Plotting, metrics, and logging helpers for the CityLearn HVAC RL project.

All plotting functions save figures to config.RESULTS_DIR and also return
the matplotlib Figure object so callers can display or further modify them.

Functions
---------
plot_training_rewards   : episode reward curve from SB3 Monitor logs
plot_kpi_comparison     : grouped bar chart comparing baseline vs RL KPIs
plot_temperature_trace  : indoor temperature vs setpoint over an episode
plot_reward_comparison  : per-step reward comparison (baseline vs RL)
save_metrics_csv        : save a KPI DataFrame to a CSV file
print_summary_table     : print a formatted comparison table to the console
"""

import os
import sys
import csv
import glob
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # use non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, filename: str) -> str:
    """Save a figure to results dir and return the full path."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, filename)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    print(f"  [plot] Saved: {path}")
    return path


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent styling to a set of axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Training reward curve
# ---------------------------------------------------------------------------

def plot_training_rewards(
    log_dir: str = config.MONITOR_LOG_DIR,
    smoothing_window: int = 3,
) -> Optional[plt.Figure]:
    """
    Plot the episode reward curve recorded by SB3's Monitor wrapper.

    SB3 Monitor writes a CSV file named ``<prefix>.monitor.csv`` under
    *log_dir*.  Each row is one episode with columns ``r`` (reward),
    ``l`` (length), and ``t`` (elapsed time).

    Parameters
    ----------
    log_dir : str
        Directory containing ``*.monitor.csv`` files.
    smoothing_window : int
        Rolling-average window size for the reward curve.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object, or None if no monitor CSV was found.
    """
    # Find monitor CSV files.
    pattern = os.path.join(log_dir, "*.monitor.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"  [plot] No monitor CSV found in '{log_dir}'. "
              "Run training first.")
        return None

    # Load and concatenate all monitor files (usually just one).
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, comment="#")
            frames.append(df)
        except Exception as exc:
            print(f"  [plot] Warning: could not read {f} ({exc}).")

    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True)
    rewards = data["r"].values
    episodes = np.arange(1, len(rewards) + 1)

    # Smooth with a rolling mean.
    if smoothing_window > 1 and len(rewards) >= smoothing_window:
        smoothed = (
            pd.Series(rewards)
            .rolling(window=smoothing_window, min_periods=1)
            .mean()
            .values
        )
    else:
        smoothed = rewards

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Raw reward")
    ax.plot(
        episodes, smoothed,
        color="steelblue", linewidth=2,
        label=f"Smoothed (window={smoothing_window})"
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    _style_axes(ax, "SAC Training — Episode Reward", "Episode", "Total Reward")
    ax.legend(fontsize=10)
    fig.tight_layout()

    _save_fig(fig, "training_rewards.png")
    return fig


# ---------------------------------------------------------------------------
# 2. KPI comparison bar chart
# ---------------------------------------------------------------------------

# Metrics to include in the comparison chart (subset of env.evaluate() output).
_KPI_METRICS_OF_INTEREST = [
    "electricity_consumption_total",
    "cost_total",
    "carbon_emissions_total",
    "discomfort_proportion",
    "discomfort_hot_proportion",
    "discomfort_cold_proportion",
]


def plot_kpi_comparison(
    baseline_kpis: pd.DataFrame,
    rl_kpis: pd.DataFrame,
    filename: str = "kpi_comparison.png",
) -> plt.Figure:
    """
    Grouped bar chart comparing KPIs of the baseline vs the RL agent.

    KPI values from CityLearn are normalised relative to the no-control
    reference (1.0 = same as doing nothing).  Values below 1.0 indicate
    improvement; a red dashed line at 1.0 marks the reference.

    Parameters
    ----------
    baseline_kpis : pd.DataFrame
        KPI table from run_baseline() (index = metric name).
    rl_kpis : pd.DataFrame
        KPI table from evaluate_sac() (index = metric name).
    filename : str
        Output file name inside config.RESULTS_DIR.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Extract the building column (first non-District column) for each table.
    def _extract_building_col(kpis: pd.DataFrame) -> pd.Series:
        """Return the first building column (not 'District') from a KPI table."""
        cols = [c for c in kpis.columns if c != "District"]
        if cols:
            return kpis[cols[0]]
        return kpis.iloc[:, 0]

    # Filter to metrics of interest; silently skip missing ones.
    def _filter_metrics(series: pd.Series) -> pd.Series:
        available = [m for m in _KPI_METRICS_OF_INTEREST if m in series.index]
        return series.loc[available]

    baseline_col = _filter_metrics(_extract_building_col(baseline_kpis))
    rl_col = _filter_metrics(_extract_building_col(rl_kpis))

    # Align indices.
    common_idx = baseline_col.index.intersection(rl_col.index)
    baseline_vals = baseline_col.loc[common_idx]
    rl_vals = rl_col.loc[common_idx]
    labels = [m.replace("_", "\n") for m in common_idx]

    x = np.arange(len(common_idx))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(common_idx) * 1.6), 5))

    bars_b = ax.bar(x - width / 2, baseline_vals, width,
                    label="Rule-Based (baseline)", color="#4C72B0", alpha=0.85)
    bars_r = ax.bar(x + width / 2, rl_vals, width,
                    label="SAC (RL agent)", color="#DD8452", alpha=0.85)

    # Reference line at 1.0 (= no-control baseline).
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2,
               label="No-control reference (1.0)")

    # Annotate bars with their values.
    for bar in list(bars_b) + list(bars_r):
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _style_axes(ax, "KPI Comparison: Baseline vs SAC Agent",
                "Metric", "Normalised Value (lower = better for most metrics)")
    ax.legend(fontsize=10)
    fig.tight_layout()

    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 3. Temperature trace
# ---------------------------------------------------------------------------

def plot_temperature_trace(
    indoor_temps: List[float],
    setpoints: List[float],
    label: str = "Agent",
    filename: Optional[str] = None,
    n_steps: Optional[int] = None,
) -> plt.Figure:
    """
    Plot indoor temperature vs setpoint over an evaluation episode.

    A shaded comfort band (±COMFORT_BAND around the setpoint) is drawn so
    it is easy to see when the agent violates comfort constraints.

    Parameters
    ----------
    indoor_temps : list of float
        Sequence of indoor temperatures (°C) recorded each timestep.
    setpoints : list of float
        Corresponding setpoint temperatures (°C).
    label : str
        Agent label for the plot title and legend (e.g. "Baseline" or "SAC").
    filename : str, optional
        Output filename. Defaults to ``temperature_trace_{label}.png``.
    n_steps : int, optional
        How many timesteps to display. Defaults to config.TEMPERATURE_TRACE_STEPS.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = n_steps or config.TEMPERATURE_TRACE_STEPS or len(indoor_temps)
    n = min(n, len(indoor_temps))

    t_in = np.array(indoor_temps[:n])
    t_sp = np.array(setpoints[:n])
    hours = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Comfort band shading.
    ax.fill_between(
        hours,
        t_sp - config.COMFORT_BAND,
        t_sp + config.COMFORT_BAND,
        alpha=0.15, color="green", label=f"Comfort band (±{config.COMFORT_BAND} °C)"
    )

    ax.plot(hours, t_sp, color="green", linestyle="--",
            linewidth=1.5, label="Setpoint")
    ax.plot(hours, t_in, color="steelblue", linewidth=1.5,
            label="Indoor temperature")

    # Highlight comfort violations.
    violation = np.abs(t_in - t_sp) > config.COMFORT_BAND
    if violation.any():
        ax.scatter(hours[violation], t_in[violation],
                   color="crimson", s=8, zorder=5, label="Comfort violation")

    _style_axes(
        ax,
        f"Indoor Temperature vs Setpoint — {label}",
        "Hour", "Temperature (°C)"
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    if filename is None:
        filename = f"temperature_trace_{label.lower().replace(' ', '_')}.png"
    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 4. Per-step reward comparison
# ---------------------------------------------------------------------------

def plot_reward_comparison(
    baseline_rewards: List[float],
    rl_rewards: List[float],
    filename: str = "reward_comparison.png",
) -> plt.Figure:
    """
    Plot baseline vs RL per-step rewards side by side.

    Uses a 7-day (168-step) rolling mean to smooth the noisy per-step signal.

    Parameters
    ----------
    baseline_rewards : list of float
    rl_rewards : list of float
    filename : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    window = 168  # 7 days × 24 hours

    def smooth(arr):
        s = pd.Series(arr).rolling(window=window, min_periods=1).mean().values
        return s

    n = min(len(baseline_rewards), len(rl_rewards))
    x = np.arange(n)
    b_smooth = smooth(baseline_rewards[:n])
    r_smooth = smooth(rl_rewards[:n])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, b_smooth, color="#4C72B0", linewidth=1.5, label="Rule-Based (baseline)")
    ax.plot(x, r_smooth, color="#DD8452", linewidth=1.5, label="SAC (RL agent)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    _style_axes(
        ax,
        f"Per-Step Reward Comparison (7-day rolling mean)",
        "Timestep (hours)", "Reward"
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 5. Metrics helpers
# ---------------------------------------------------------------------------

def save_metrics_csv(kpis: pd.DataFrame, path: str) -> None:
    """
    Save a KPI DataFrame to a CSV file.

    Parameters
    ----------
    kpis : pd.DataFrame
        KPI pivot table from env.evaluate().
    path : str
        Full output path (including filename and .csv extension).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    kpis.to_csv(path)
    print(f"  [metrics] Saved: {path}")


def print_summary_table(
    baseline_kpis: pd.DataFrame,
    rl_kpis: pd.DataFrame,
) -> None:
    """
    Print a side-by-side comparison table of key KPIs to the console.

    Parameters
    ----------
    baseline_kpis : pd.DataFrame
    rl_kpis : pd.DataFrame
    """
    metrics = _KPI_METRICS_OF_INTEREST

    def _get_col(kpis: pd.DataFrame) -> pd.Series:
        cols = [c for c in kpis.columns if c != "District"]
        col = cols[0] if cols else kpis.columns[0]
        return kpis[col]

    baseline_col = _get_col(baseline_kpis)
    rl_col = _get_col(rl_kpis)

    rows = []
    for m in metrics:
        b_val = baseline_col.get(m, float("nan"))
        r_val = rl_col.get(m, float("nan"))
        try:
            delta = r_val - b_val
            delta_str = f"{delta:+.4f}"
        except TypeError:
            delta_str = "N/A"
        rows.append((m, f"{b_val:.4f}" if not np.isnan(b_val) else "N/A",
                     f"{r_val:.4f}" if not np.isnan(r_val) else "N/A",
                     delta_str))

    header = f"{'Metric':<40} {'Baseline':>10} {'SAC':>10} {'Delta':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for m, b, r, d in rows:
        # Highlight improvements (negative delta = better for most metrics).
        marker = " <-- improved" if d.startswith("-") else ""
        print(f"{m:<40} {b:>10} {r:>10} {d:>10}{marker}")
    print(sep + "\n")
    print("  Note: All values are normalised relative to the no-control reference.")
    print("        Values < 1.0 = improvement over doing nothing.\n")
