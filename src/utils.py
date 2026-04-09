"""
utils.py
--------
Plotting, metrics, and logging helpers for the CityLearn HVAC RL project.

All plotting functions accept a dict of ``{label: data}`` so they can render
any number of agents (RBC, SAC, PPO, …) side-by-side without code changes.

Functions
---------
plot_training_rewards   : episode reward curves from SB3 Monitor logs
plot_kpi_comparison     : grouped bar chart comparing all agent KPIs
plot_temperature_trace  : indoor temperature vs setpoint over an episode
plot_reward_comparison  : per-step reward comparison across agents
save_metrics_csv        : save a KPI DataFrame to a CSV file
print_summary_table     : print a formatted comparison table to the console
"""

import os
import sys
import glob
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config

# Consistent colour palette for up to 5 agents.
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# KPIs to show in bar charts and summary tables.
_KPI_METRICS_OF_INTEREST = [
    "electricity_consumption_total",
    "cost_total",
    "carbon_emissions_total",
    "discomfort_proportion",
    "discomfort_hot_proportion",
    "discomfort_cold_proportion",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, filename: str) -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, filename)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    print(f"  [plot] Saved: {path}")
    return path


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_monitor_rewards(log_dir: str) -> Optional[np.ndarray]:
    """Load episode rewards from all SB3 Monitor CSVs in *log_dir*."""
    files = sorted(glob.glob(os.path.join(log_dir, "*.monitor.csv")))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, comment="#"))
        except Exception:
            pass
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)["r"].values


def _extract_building_col(kpis: pd.DataFrame) -> pd.Series:
    """Return the first non-District column from a KPI pivot table."""
    cols = [c for c in kpis.columns if c != "District"]
    return kpis[cols[0]] if cols else kpis.iloc[:, 0]


# ---------------------------------------------------------------------------
# 1. Training reward curves
# ---------------------------------------------------------------------------

def plot_training_rewards(
    log_dirs: Optional[Dict[str, str]] = None,
    smoothing_window: int = 3,
    filename: str = "training_rewards.png",
) -> Optional[plt.Figure]:
    """
    Plot episode reward curves recorded by SB3's Monitor wrapper.

    Parameters
    ----------
    log_dirs : dict {label: log_dir_path}, optional
        If None, defaults to ``{"SAC": config.SAC_MONITOR_LOG_DIR,
        "PPO": config.PPO_MONITOR_LOG_DIR}`` (skips any that are missing).
    smoothing_window : int
        Rolling-average window size.
    filename : str
        Output filename inside config.RESULTS_DIR.
    """
    if log_dirs is None:
        log_dirs = {
            "SAC": config.SAC_MONITOR_LOG_DIR,
            "PPO": config.PPO_MONITOR_LOG_DIR,
        }

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = 0

    for (label, log_dir), colour in zip(log_dirs.items(), _PALETTE):
        rewards = _load_monitor_rewards(log_dir)
        if rewards is None:
            print(f"  [plot] No monitor CSV for '{label}' in '{log_dir}' — skipping.")
            continue

        episodes = np.arange(1, len(rewards) + 1)
        smoothed = (
            pd.Series(rewards)
            .rolling(window=smoothing_window, min_periods=1)
            .mean()
            .values
        )

        ax.plot(episodes, rewards, alpha=0.25, color=colour)
        ax.plot(episodes, smoothed, color=colour, linewidth=2, label=label)
        plotted += 1

    if plotted == 0:
        print("  [plot] No training data found — skipping reward curve.")
        plt.close(fig)
        return None

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    _style_axes(ax, "Training — Episode Reward", "Episode", "Total Reward")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 2. KPI comparison bar chart
# ---------------------------------------------------------------------------

def plot_kpi_comparison(
    agents: Dict[str, pd.DataFrame],
    filename: str = "kpi_comparison.png",
) -> plt.Figure:
    """
    Grouped bar chart comparing KPIs across multiple agents.

    Parameters
    ----------
    agents : dict {label: kpi_DataFrame}
        e.g. ``{"RBC": baseline_kpis, "SAC": sac_kpis, "PPO": ppo_kpis}``
    filename : str
        Output filename inside config.RESULTS_DIR.
    """
    # Build a combined DataFrame: rows = metrics, columns = agent labels.
    combined = {}
    for label, kpis in agents.items():
        series = _extract_building_col(kpis)
        available = [m for m in _KPI_METRICS_OF_INTEREST if m in series.index]
        combined[label] = series.loc[available]

    df = pd.DataFrame(combined)
    df = df.dropna(how="all")

    if df.empty:
        print("  [plot] No common KPI metrics found — skipping bar chart.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    labels = [m.replace("_", "\n") for m in df.index]
    n_agents = len(df.columns)
    n_metrics = len(df)
    width = 0.8 / n_agents
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.8), 5))

    for i, (col, colour) in enumerate(zip(df.columns, _PALETTE)):
        vals = df[col].values
        offset = (i - n_agents / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=col, color=colour, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2,
               label="No-control reference (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _style_axes(ax, "KPI Comparison — All Agents",
                "Metric", "Normalised Value (lower = better for most metrics)")
    ax.legend(fontsize=9)
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
    Plot indoor temperature vs setpoint for one agent's evaluation episode.

    Parameters
    ----------
    indoor_temps : list of float
    setpoints : list of float
    label : str
        Agent label for title and legend.
    filename : str, optional
    n_steps : int, optional
        Number of timesteps to display. Defaults to config.TEMPERATURE_TRACE_STEPS.
    """
    n = n_steps or config.TEMPERATURE_TRACE_STEPS or len(indoor_temps)
    n = min(n, len(indoor_temps))
    t_in = np.array(indoor_temps[:n])
    hours = np.arange(n)

    # Setpoints are optional — LSTMDynamicsBuilding may not expose them.
    has_setpoints = len(setpoints) >= n
    t_sp = np.array(setpoints[:n]) if has_setpoints else None

    fig, ax = plt.subplots(figsize=(12, 4))

    if has_setpoints and t_sp is not None:
        ax.fill_between(
            hours, t_sp - config.COMFORT_BAND, t_sp + config.COMFORT_BAND,
            alpha=0.15, color="green",
            label=f"Comfort band (±{config.COMFORT_BAND} °C)",
        )
        ax.plot(hours, t_sp, color="green", linestyle="--", linewidth=1.5, label="Setpoint")
        violation = np.abs(t_in - t_sp) > config.COMFORT_BAND
        if violation.any():
            ax.scatter(hours[violation], t_in[violation],
                       color="crimson", s=8, zorder=5, label="Comfort violation")

    ax.plot(hours, t_in, color="steelblue", linewidth=1.5, label="Indoor temperature")

    _style_axes(ax, f"Indoor Temperature vs Setpoint — {label}",
                "Hour", "Temperature (°C)")
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
    agents: Optional[Dict[str, List[float]]] = None,
    filename: str = "reward_comparison.png",
    # Legacy positional args kept for backward compatibility
    baseline_rewards: Optional[List[float]] = None,
    rl_rewards: Optional[List[float]] = None,
) -> plt.Figure:
    """
    Per-step reward comparison with a 7-day rolling mean.

    Parameters
    ----------
    agents : dict {label: rewards_list}, optional
        e.g. ``{"RBC": [...], "SAC": [...], "PPO": [...]}``
        If None, falls back to the legacy ``baseline_rewards`` / ``rl_rewards``
        keyword arguments for backward compatibility.
    filename : str
    """
    # Legacy two-argument form
    if agents is None:
        if baseline_rewards is None or rl_rewards is None:
            print("  [plot] No reward data provided — skipping comparison.")
            fig, ax = plt.subplots()
            return fig
        agents = {"Rule-Based (RBC)": baseline_rewards, "SAC": rl_rewards}

    window = 168   # 7 days × 24 hours

    def smooth(arr):
        return pd.Series(arr).rolling(window=window, min_periods=1).mean().values

    # Trim all series to the shortest length for fair comparison.
    n = min(len(v) for v in agents.values())
    hours = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))
    for (label, rewards), colour in zip(agents.items(), _PALETTE):
        ax.plot(hours, smooth(rewards[:n]), color=colour, linewidth=1.5, label=label)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    _style_axes(ax, "Per-Step Reward Comparison (7-day rolling mean)",
                "Timestep (hours)", "Reward")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save_fig(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# 5. Metrics helpers
# ---------------------------------------------------------------------------

def save_metrics_csv(kpis: pd.DataFrame, path: str) -> None:
    """Save a KPI DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    kpis.to_csv(path)
    print(f"  [metrics] Saved: {path}")


def print_summary_table(agents: Dict[str, pd.DataFrame]) -> None:
    """
    Print a side-by-side KPI comparison table for all agents.

    Parameters
    ----------
    agents : dict {label: kpi_DataFrame}
        e.g. ``{"RBC": baseline_kpis, "SAC": sac_kpis, "PPO": ppo_kpis}``
    """
    metrics = _KPI_METRICS_OF_INTEREST
    col_width = 11

    # Build a dict of {label: Series} for the building column.
    cols = {}
    for label, kpis in agents.items():
        cols[label] = _extract_building_col(kpis)

    # Header.
    header = f"{'Metric':<40}" + "".join(f"{lbl:>{col_width}}" for lbl in cols)
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for m in metrics:
        row = f"{m:<40}"
        vals = {}
        for label, series in cols.items():
            v = series.get(m, float("nan"))
            vals[label] = v
            row += f"{v:>{col_width}.4f}" if not (isinstance(v, float) and np.isnan(v)) else f"{'N/A':>{col_width}}"

        # Mark the best (lowest) value among agents.
        numeric = {k: v for k, v in vals.items() if isinstance(v, float) and not np.isnan(v)}
        if numeric:
            best_label = min(numeric, key=numeric.get)
            row += f"  <- best: {best_label}"

        print(row)

    print(sep)
    print("\n  Note: all values normalised vs. no-control reference (1.0 = no improvement).")
    print("        Values < 1.0 = better than doing nothing.\n")
