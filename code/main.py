# Importing Libraries
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import (OptimizeResult, differential_evolution, minimize)
from scipy.spatial import cKDTree

# Configuration Constants
CSV_PATH = r"dataset.csv"

T_MIN: float = 6.0
T_MAX: float = 60.0
Y_OFFSET: float = 42.0  # fixed additive constant in y(t)

# Number of points used to densely sample the candidate curve during optimization (N >= 20000)
N_CURVE_OPTIMIZATION: int = 3000
N_CURVE_FINAL: int = 50000

# Parameter bounds: (theta_degrees, M, X)
PARAM_BOUNDS: list[tuple[float, float]] = [
    (0.0, 50.0),  # theta in degrees
    (-0.05, 0.05),  # M
    (0.0, 100.0),  # X
]

# Differential Evolution hyperparameters
DE_STRATEGY: str = "best1bin"
DE_POPSIZE: int = 10
DE_MAXITER: int = 30
DE_MUTATION: tuple[float, float] = (0.5, 1.0)
DE_RECOMBINATION: float = 0.7
DE_POLISH: bool = False
DE_WORKERS: int = 1
DE_SEED: int = 42
DE_TOL: float = 1e-6
DE_ATOL: float = 1e-8

RNG_SEED: int = 42

# Data Containers
@dataclass
class OptimizationDiagnostics:
    """Container for diagnostics collected during the optimization run."""

    de_result: OptimizeResult | None = None
    de_params: np.ndarray | None = None
    de_objective: float | None = None
    de_runtime_s: float | None = None
    de_nfev: int | None = None

    lbfgsb_result: OptimizeResult | None = None
    lbfgsb_params: np.ndarray | None = None
    lbfgsb_objective: float | None = None
    lbfgsb_runtime_s: float | None = None
    lbfgsb_nfev: int | None = None

    total_runtime_s: float | None = None
    improvement_pct: float | None = None


@dataclass
class ErrorMetrics:
    """Container for the full suite of nearest-neighbor error metrics."""

    mean_l1: float
    median_l1: float
    max_l1: float
    mean_euclidean: float
    median_euclidean: float
    max_euclidean: float
    rmse: float
    mae: float
    std_residual: float
    l1_residuals: np.ndarray = field(repr = False)
    euclidean_residuals: np.ndarray = field(repr = False)

# Core Model Functions
def generate_curve(
    theta_deg: float, M: float, X: float, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dense sample of the parametric curve.

    Parameters:
    theta_deg:
        Rotation angle in degrees. Converted internally to radians.
    M:
        Exponential envelope growth/decay rate.
    X:
        Horizontal translation offset.
    num_points:
        Number of samples to draw uniformly over t in [T_MIN, T_MAX].

    Returns:
    (x_curve, y_curve):
        1D NumPy arrays of shape (num_points,) giving the curve's x and y
        coordinates at each sampled t value.
    """
    theta_rad = np.radians(theta_deg)
    t = np.linspace(T_MIN, T_MAX, num_points)

    envelope = np.exp(M * np.abs(t)) * np.sin(0.3 * t)

    x_curve = t * np.cos(theta_rad) - envelope * np.sin(theta_rad) + X
    y_curve = Y_OFFSET + t * np.sin(theta_rad) + envelope * np.cos(theta_rad)

    return x_curve, y_curve


class _MeanNearestNeighborL1Objective:
    """Picklable callable implementing the mean nearest-neighbor L1 objective.

    Implemented as a module-level class (rather than a closure) so that it
    remains picklable for ``differential_evolution(..., workers=-1)``, which
    relies on ``multiprocessing`` to evaluate population members in parallel.
    """

    __slots__ = ("observed_points", "n_obs", "num_curve_points")

    def __init__(self, x_obs: np.ndarray, y_obs: np.ndarray, num_curve_points: int) -> None:
        self.observed_points = np.column_stack([x_obs, y_obs])
        self.n_obs = self.observed_points.shape[0]
        self.num_curve_points = num_curve_points

    def __call__(self, params: np.ndarray) -> float:
        theta_deg, M, X = params
        x_curve, y_curve = generate_curve(theta_deg, M, X, self.num_curve_points)
        curve_points = np.column_stack([x_curve, y_curve])

        # cKDTree defaults to Euclidean (p = 2) queries, we explicitly request p = 1 (Manhattan / L1) nearest-neighbor distances.
        tree = cKDTree(curve_points)
        l1_distances, _ = tree.query(self.observed_points, k=1, p=1)

        return float(np.sum(l1_distances) / self.n_obs)


def build_objective(x_obs: np.ndarray, y_obs: np.ndarray, num_curve_points: int) -> Callable[[np.ndarray], float]:
    """Construct the mean nearest-neighbor L1 distance objective function."""
    return _MeanNearestNeighborL1Objective(x_obs, y_obs, num_curve_points)

# Data Loading
def load_observed_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the observed (x, y) point cloud from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path!r}")

    df = pd.read_csv(path)

    required_columns = {"x", "y"}
    missing = required_columns - set(df.columns.str.strip())
    if missing:
        raise ValueError(f"CSV file is missing required column(s): {sorted(missing)}. " f"Found columns: {list(df.columns)}")

    # Dropping any rows with missing/invalid values defensively.
    df = df.dropna(subset=["x", "y"])
    if df.empty:
        raise ValueError("CSV file contains no valid (x, y) rows after cleaning.")

    x_obs = df["x"].to_numpy(dtype=np.float64)
    y_obs = df["y"].to_numpy(dtype=np.float64)

    return x_obs, y_obs

# Optimization Pipeline
def run_differential_evolution(objective: Callable[[np.ndarray], float]) -> tuple[OptimizeResult, float]:
    """Run the global-search (Stage 1) optimization via Differential Evolution."""
    start = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds = PARAM_BOUNDS,
        strategy = DE_STRATEGY,
        popsize = DE_POPSIZE,
        maxiter = DE_MAXITER,
        mutation = DE_MUTATION,
        recombination = DE_RECOMBINATION,
        polish = DE_POLISH,
        workers = DE_WORKERS,
        seed = DE_SEED,
        tol = DE_TOL,
        atol = DE_ATOL,
        updating = "deferred",  # required for workers=-1 (parallel) correctness
    )
    runtime = time.perf_counter() - start
    return result, runtime


def run_lbfgsb_refinement(objective: Callable[[np.ndarray], float], x0: np.ndarray) -> tuple[OptimizeResult, float]:
    """Run the local-refinement (Stage 2) optimization via L-BFGS-B."""
    start = time.perf_counter()
    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=PARAM_BOUNDS,
        options={"maxiter": 2000, "maxfun": 20000, "ftol": 1e-12, "gtol": 1e-10},
    )
    runtime = time.perf_counter() - start
    return result, runtime


def optimize_parameters(x_obs: np.ndarray, y_obs: np.ndarray) -> tuple[np.ndarray, OptimizationDiagnostics]:
    """Run the full two-stage hybrid optimization pipeline."""
    diagnostics = OptimizationDiagnostics()
    objective = build_objective(x_obs, y_obs, N_CURVE_OPTIMIZATION)

    total_start = time.perf_counter()

    # ---------------- Stage 1: Differential Evolution ----------------
    print("Stage 1: Differential Evolution (global search)")
    de_result, de_runtime = run_differential_evolution(objective)

    diagnostics.de_result = de_result
    diagnostics.de_params = de_result.x
    diagnostics.de_objective = float(de_result.fun)
    diagnostics.de_runtime_s = de_runtime
    diagnostics.de_nfev = de_result.nfev

    print(f" DE complete: objective={de_result.fun:.6f}, " f"runtime={de_runtime:.2f}s, nfev={de_result.nfev}")

    # ---------------- Stage 2: L-BFGS-B refinement --------------------
    print("Stage 2: L-BFGS-B (local refinement)")
    lbfgsb_result, lbfgsb_runtime = run_lbfgsb_refinement(objective, de_result.x)

    diagnostics.lbfgsb_result = lbfgsb_result
    diagnostics.lbfgsb_params = lbfgsb_result.x
    diagnostics.lbfgsb_objective = float(lbfgsb_result.fun)
    diagnostics.lbfgsb_runtime_s = lbfgsb_runtime
    diagnostics.lbfgsb_nfev = lbfgsb_result.nfev

    print(f"  L-BFGS-B complete: objective={lbfgsb_result.fun:.6f}, " f"runtime={lbfgsb_runtime:.2f}s, nfev={lbfgsb_result.nfev}")

    diagnostics.total_runtime_s = time.perf_counter() - total_start

    if diagnostics.de_objective and diagnostics.de_objective > 0:
        diagnostics.improvement_pct = (diagnostics.de_objective - diagnostics.lbfgsb_objective) / diagnostics.de_objective * 100.0
    else:
        diagnostics.improvement_pct = 0.0

    # L-BFGS-B is a local refiner: only accepting its result if it did not diverge to a worse objective value than the DE starting point
    # (a defensive safeguard; in well-behaved cases this branch is not hit).
    if diagnostics.lbfgsb_objective <= diagnostics.de_objective:
        best_params = lbfgsb_result.x
    else:
        warnings.warn("L-BFGS-B refinement did not improve on the Differential " "Evolution solution; falling back to the DE result.", stacklevel = 2)
        best_params = de_result.x

    return best_params, diagnostics

# Error Metrics
def compute_error_metrics(x_obs: np.ndarray, y_obs: np.ndarray, params: np.ndarray, num_curve_points: int = N_CURVE_FINAL) -> ErrorMetrics:
    """Compute the full suite of nearest-neighbor fit-quality metrics."""
    theta_deg, M, X = params
    x_curve, y_curve = generate_curve(theta_deg, M, X, num_curve_points)

    observed_points = np.column_stack([x_obs, y_obs])
    curve_points = np.column_stack([x_curve, y_curve])
    tree = cKDTree(curve_points)

    l1_distances, _ = tree.query(observed_points, k=1, p=1)
    euclidean_distances, _ = tree.query(observed_points, k=1, p=2)

    mean_l1 = float(np.mean(l1_distances))
    median_l1 = float(np.median(l1_distances))
    max_l1 = float(np.max(l1_distances))

    mean_euclidean = float(np.mean(euclidean_distances))
    median_euclidean = float(np.median(euclidean_distances))
    max_euclidean = float(np.max(euclidean_distances))

    rmse = float(np.sqrt(np.mean(euclidean_distances**2)))
    mae = float(np.mean(np.abs(euclidean_distances)))
    std_residual = float(np.std(euclidean_distances))

    return ErrorMetrics(
        mean_l1 = mean_l1,
        median_l1 = median_l1,
        max_l1 = max_l1,
        mean_euclidean = mean_euclidean,
        median_euclidean = median_euclidean,
        max_euclidean = max_euclidean,
        rmse = rmse,
        mae = mae,
        std_residual = std_residual,
        l1_residuals = l1_distances,
        euclidean_residuals = euclidean_distances,
    )

# Visualizations
def plot_observed_points(x_obs: np.ndarray, y_obs: np.ndarray) -> None:
    """Observed points only."""
    fig, ax = plt.subplots(figsize = (9, 7), dpi=150)
    ax.scatter(x_obs, y_obs, s = 6, c = "steelblue", alpha = 0.6, edgecolors = "none")
    ax.set_title("Observed Data Points", fontsize = 14, fontweight = "bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha = 0.4)
    fig.tight_layout()
    plt.show()


def plot_recovered_curve(params: np.ndarray) -> None:
    """Recovered curve only."""
    theta_deg, M, X = params
    x_curve, y_curve = generate_curve(theta_deg, M, X, N_CURVE_FINAL)

    fig, ax = plt.subplots(figsize = (9, 7), dpi = 150)
    ax.plot(x_curve, y_curve, c = "crimson", linewidth = 1.2)
    ax.set_title(f"Recovered Curve  (theta = {theta_deg:.4f} deg, M = {M:.6f}, X = {X:.4f})", fontsize = 13, fontweight = "bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha = 0.4)
    fig.tight_layout()
    plt.show()


def plot_overlay(x_obs: np.ndarray, y_obs: np.ndarray, params: np.ndarray) -> None:
    """Observed points + recovered curve overlay."""
    theta_deg, M, X = params
    x_curve, y_curve = generate_curve(theta_deg, M, X, N_CURVE_FINAL)

    fig, ax = plt.subplots(figsize = (9, 7), dpi = 150)
    ax.scatter(x_obs, y_obs, s = 6, c = "steelblue", alpha = 0.5, edgecolors = "none", label = "Observed points")
    ax.plot(x_curve, y_curve, c = "crimson", linewidth = 1.2, label = "Recovered curve")
    ax.set_title("Observed Points + Recovered Curve Overlay", fontsize=14, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend(loc = "best")
    ax.grid(True, linestyle = "--", alpha = 0.4)
    fig.tight_layout()
    plt.show()


def plot_residual_histogram(metrics: ErrorMetrics) -> None:
    """Histogram of nearest-neighbor residuals."""
    fig, axes = plt.subplots(1, 2, figsize = (13, 5), dpi = 150)

    axes[0].hist(metrics.l1_residuals, bins = 50, color = "teal", edgecolor = "black", alpha = 0.75)
    axes[0].axvline(metrics.mean_l1, color = "red", linestyle = "--", label = f"Mean = {metrics.mean_l1:.4f}")
    axes[0].axvline(metrics.median_l1, color = "orange", linestyle = "--", label = f"Median = {metrics.median_l1:.4f}")
    axes[0].set_title("Nearest-Neighbor L1 Residuals")
    axes[0].set_xlabel("L1 distance")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].hist(metrics.euclidean_residuals, bins = 50, color = "purple", edgecolor = "black", alpha = 0.75)
    axes[1].axvline(metrics.mean_euclidean, color = "red", linestyle = "--", label = f"Mean = {metrics.mean_euclidean:.4f}")
    axes[1].axvline(metrics.median_euclidean, color = "orange", linestyle = "--", label = f"Median = {metrics.median_euclidean:.4f}")
    axes[1].set_title("Nearest-Neighbor Euclidean Residuals")
    axes[1].set_xlabel("Euclidean distance")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, linestyle = "--", alpha = 0.4)

    fig.suptitle("Residual Distributions", fontsize = 14, fontweight = "bold")
    fig.tight_layout()
    plt.show()


def plot_residual_scatter(x_obs: np.ndarray, y_obs: np.ndarray, metrics: ErrorMetrics) -> None:
    """Residual magnitude scatter plot."""
    fig, ax = plt.subplots(figsize = (9, 7), dpi = 150)
    scatter = ax.scatter(x_obs, y_obs, c = metrics.euclidean_residuals, cmap = "viridis", s = 10, edgecolors = "none")
    cbar = fig.colorbar(scatter, ax = ax)
    cbar.set_label("Euclidean residual magnitude")
    ax.set_title("Residual Magnitude Scatter Plot", fontsize = 14, fontweight = "bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linestyle = "--", alpha = 0.3)
    fig.tight_layout()
    plt.show()


def plot_zoomed_overlay(x_obs: np.ndarray, y_obs: np.ndarray, params: np.ndarray) -> None:
    """Zoomed overlay view (centered on the observed data's midpoint)."""
    theta_deg, M, X = params
    x_curve, y_curve = generate_curve(theta_deg, M, X, N_CURVE_FINAL)

    x_center = float(np.median(x_obs))
    y_center = float(np.median(y_obs))
    x_span = float(np.ptp(x_obs))
    y_span = float(np.ptp(y_obs))
    zoom_half_width = 0.15 * max(x_span, y_span, 1.0)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    ax.scatter(x_obs, y_obs, s = 14, c = "steelblue", alpha = 0.7, edgecolors = "none", label = "Observed points")
    ax.plot(x_curve, y_curve, c = "crimson", linewidth = 1.5, label = "Recovered curve")
    ax.set_xlim(x_center - zoom_half_width, x_center + zoom_half_width)
    ax.set_ylim(y_center - zoom_half_width, y_center + zoom_half_width)
    ax.set_title("Zoomed Overlay View", fontsize = 14, fontweight = "bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend(loc = "best")
    ax.grid(True, linestyle = "--", alpha = 0.4)
    fig.tight_layout()
    plt.show()


def plot_de_vs_refined(de_params: np.ndarray, refined_params: np.ndarray) -> None:
    """Curve from DE solution vs curve from final refined solution."""
    x_de, y_de = generate_curve(*de_params, N_CURVE_FINAL)
    x_refined, y_refined = generate_curve(*refined_params, N_CURVE_FINAL)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    ax.plot(x_de, y_de, c = "darkorange", linewidth = 1.4, linestyle = "--", label = "DE solution curve")
    ax.plot(x_refined, y_refined, c = "crimson", linewidth = 1.4, label = "L-BFGS-B refined curve")
    ax.set_title("DE Solution vs. L-BFGS-B Refined Solution", fontsize=14, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend(loc = "best")
    ax.grid(True, linestyle = "--", alpha = 0.4)
    fig.tight_layout()
    plt.show()

# Final Report
def print_final_report(params: np.ndarray, objective_value: float, metrics: ErrorMetrics, diagnostics: OptimizationDiagnostics) -> None:
    """Print the final results."""
    theta_deg, M, X = params

    print("\n" + "=" * 50)
    print("FINAL PARAMETERS")
    print("=" * 50)
    print(f"\nTheta = {theta_deg:.6f}")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")

    print("\n" + "=" * 50)
    print("FITNESS")
    print("=" * 50)
    print(f"\nObjective Value = {objective_value:.6f}")

    print("\n" + "=" * 50)
    print("ERROR METRICS")
    print("=" * 50)
    print(f"\nMean L1 Distance        = {metrics.mean_l1:.6f}")
    print(f"Median L1 Distance      = {metrics.median_l1:.6f}")
    print(f"Maximum L1 Distance     = {metrics.max_l1:.6f}")
    print(f"\nMean Euclidean Distance = {metrics.mean_euclidean:.6f}")
    print(f"Median Euclidean Distance = {metrics.median_euclidean:.6f}")
    print(f"Maximum Euclidean Distance = {metrics.max_euclidean:.6f}")
    print(f"\nRMSE = {metrics.rmse:.6f}")
    print(f"MAE  = {metrics.mae:.6f}")
    print(f"STD  = {metrics.std_residual:.6f}")

    print("\n" + "=" * 50)
    print("RUNTIME")
    print("=" * 50)
    print(f"\nDE Runtime       = {diagnostics.de_runtime_s:.4f} s")
    print(f"L-BFGS-B Runtime = {diagnostics.lbfgsb_runtime_s:.4f} s")
    print(f"Total Runtime    = {diagnostics.total_runtime_s:.4f} s")

    print("\n" + "=" * 50)
    print("OPTIMIZER STATUS")
    print("=" * 50)
    print(f"\n--- Differential Evolution ---")
    print(f"DE Objective Value = {diagnostics.de_objective:.6f}")
    print(f"DE Parameters      = theta={diagnostics.de_params[0]:.6f}, " f"M={diagnostics.de_params[1]:.6f}, X={diagnostics.de_params[2]:.6f}")
    print(f"DE Success         = {diagnostics.de_result.success}")
    print(f"DE Message         = {diagnostics.de_result.message}")
    print(f"DE Function Evals  = {diagnostics.de_nfev}")
    print(f"DE Iterations      = {diagnostics.de_result.nit}")

    print(f"\n--- L-BFGS-B Refinement ---")
    print(f"L-BFGS-B Objective Value = {diagnostics.lbfgsb_objective:.6f}")
    print(f"L-BFGS-B Parameters      = theta={diagnostics.lbfgsb_params[0]:.6f}, " f"M={diagnostics.lbfgsb_params[1]:.6f}, X={diagnostics.lbfgsb_params[2]:.6f}")
    print(f"L-BFGS-B Success         = {diagnostics.lbfgsb_result.success}")
    print(f"L-BFGS-B Message         = {diagnostics.lbfgsb_result.message}")
    print(f"L-BFGS-B Function Evals  = {diagnostics.lbfgsb_nfev}")
    print(f"\nImprovement (DE -> L-BFGS-B) = {diagnostics.improvement_pct:.4f}%")
    print("=" * 50 + "\n")

# Running Pipeline
def main():
    np.random.seed(RNG_SEED)

    # Loading the dataset
    x_obs, y_obs = load_observed_data(CSV_PATH)
    print(f"Loaded {len(x_obs)} observed points from {CSV_PATH!r}")

    # Running the two-stage hybrid optimization (Differential Evolution + L-BFGS-B).
    final_params, diagnostics = optimize_parameters(x_obs, y_obs)

    # Computing final error metrics at high curve resolution.
    metrics = compute_error_metrics(x_obs, y_obs, final_params, N_CURVE_FINAL)

    # Plotting Graphs
    plot_observed_points(x_obs, y_obs)
    plot_recovered_curve(final_params)
    plot_overlay(x_obs, y_obs, final_params)
    plot_residual_histogram(metrics)
    plot_residual_scatter(x_obs, y_obs, metrics)
    plot_zoomed_overlay(x_obs, y_obs, final_params)
    plot_de_vs_refined(diagnostics.de_params, final_params)

    # Printing final reports
    print_final_report(params=final_params, objective_value = diagnostics.lbfgsb_objective, metrics = metrics, diagnostics = diagnostics)

if __name__ == "__main__":
    main()