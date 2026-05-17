"""
Statistical analysis for the demographic simulation.

It runs multiple independent simulations, aggregates yearly metrics with
confidence intervals, and exports the results to CSV.

Main metrics: population, births, deaths, couples, men, women, breakups.
For each metric and year: mean, std, min, max, median, P5, P95, CI95.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, TypedDict

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

YEARS_PER_SIMULATION = 100
DEFAULT_NUM_RUNS = 30
DEFAULT_OUTPUT_FILE = Path("simulation_statistics.csv")
DEFAULT_METRICS = ("population", "births", "deaths", "couples", "men", "women", "breakups")


class YearlySimulationRecord(TypedDict):
    year: int
    population: int
    men: int
    women: int
    births: int
    deaths: int
    couples: int
    breakups: int


@dataclass(slots=True)
class SimulationRunResult:
    """Stores the result of one simulation run."""

    run_id: int
    seed: Optional[int]
    records: Optional[list[YearlySimulationRecord]] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.records is not None and self.error is None


@dataclass(slots=True)
class _RunnerRegistry:
    runner: Optional[Callable[[Optional[int]], Sequence[Mapping[str, int | float]]]] = None


_runner_registry = _RunnerRegistry()


def register_simulation_runner(
    runner: Callable[[Optional[int]], Sequence[Mapping[str, int | float]]],
) -> None:
    """Set the callable that actually runs the simulator."""
    _runner_registry.runner = runner


def run_single_simulation(seed: Optional[int] = None) -> list[YearlySimulationRecord]:
    """Run one simulation through the registered runner.

    Call register_simulation_runner() first. The runner should return a
    sequence of 100 dictionaries, one per year.
    """
    runner = _runner_registry.runner
    if runner is None:
        raise NotImplementedError(
            "No simulation runner registered. Call register_simulation_runner() first."
        )

    assert runner is not None
    raw_records = runner(seed)
    return _validate_and_normalize_records(raw_records)


def _validate_and_normalize_records(
    records: Sequence[Mapping[str, int | float]],
) -> list[YearlySimulationRecord]:
    """Check a run and normalize it into the expected structure."""

    if len(records) != YEARS_PER_SIMULATION:
        raise ValueError(
            f"Each simulation must return {YEARS_PER_SIMULATION} yearly records. Got {len(records)}."
        )

    normalized: list[YearlySimulationRecord] = []
    required_keys = (
        "year", "population", "men", "women",
        "births", "deaths", "couples", "breakups",
    )

    for expected_year, record in enumerate(records):
        missing_keys = [key for key in required_keys if key not in record]
        if missing_keys:
            raise ValueError(
                f"Year {expected_year} record is missing keys: {', '.join(missing_keys)}"
            )

        year_value = int(record["year"])
        if year_value != expected_year:
            raise ValueError(
                f"Invalid year ordering. Expected year {expected_year}, got {year_value}."
            )

        normalized.append(
            {
                "year": year_value,
                "population": int(record["population"]),
                "men": int(record["men"]),
                "women": int(record["women"]),
                "births": int(record["births"]),
                "deaths": int(record["deaths"]),
                "couples": int(record["couples"]),
                "breakups": int(record["breakups"]),
            }
        )

    return normalized


def _derive_run_seed(seed: Optional[int], run_index: int) -> Optional[int]:
    """Derive a repeatable seed for each run."""
    if seed is None:
        return None
    generator = np.random.default_rng(seed)
    derived_seeds = generator.integers(0, np.iinfo(np.int32).max, size=run_index + 1)
    return int(derived_seeds[-1])


def run_multiple_simulations(
    num_runs: int = DEFAULT_NUM_RUNS,
    seed: Optional[int] = None,
) -> list[SimulationRunResult]:
    """Run several simulations and keep failures without stopping everything."""

    results: list[SimulationRunResult] = []

    for run_index in range(num_runs):
        run_seed = _derive_run_seed(seed, run_index)
        try:
            records = run_single_simulation(seed=run_seed)
            results.append(
                SimulationRunResult(
                    run_id=run_index + 1,
                    seed=run_seed,
                    records=records,
                )
            )
        except (
            NotImplementedError, ValueError, RuntimeError, TypeError,
            KeyError, IndexError, OSError, AssertionError, ArithmeticError,
        ) as exc:
            logging.exception("Run %d failed", run_index + 1)
            results.append(
                SimulationRunResult(
                    run_id=run_index + 1,
                    seed=run_seed,
                    records=None,
                    error=str(exc),
                )
            )

    successful_runs = sum(1 for result in results if result.succeeded)
    logging.info("Completed %d/%d simulation runs successfully", successful_runs, num_runs)
    return results


def compute_confidence_interval(mean: float, std: float, n: int) -> tuple[float, float]:
    """Compute the 95% confidence interval for the sample mean."""
    if n <= 1 or not math.isfinite(mean) or not math.isfinite(std):
        return mean, mean
    margin_of_error = 1.96 * (std / math.sqrt(n))
    return mean - margin_of_error, mean + margin_of_error


def aggregate_statistics(
    results: Sequence[SimulationRunResult],
    target_metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Combine yearly statistics from the runs that completed successfully."""

    successful_runs = [result for result in results if result.records is not None]
    if not successful_runs:
        return pd.DataFrame()

    flat_records: list[YearlySimulationRecord] = []
    for result in successful_runs:
        assert result.records is not None
        for record in result.records:
            flat_records.append(record)

    df = pd.DataFrame.from_records(flat_records)
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int]] = []
    successful_run_count = len(successful_runs)

    for year in range(YEARS_PER_SIMULATION):
        year_df = df[df["year"] == year]
        row: dict[str, float | int] = {
            "year": year,
            "successful_runs": successful_run_count,
        }

        for metric in target_metrics:
            if metric not in year_df.columns:
                continue

            values = [float(value) for value in year_df[metric].tolist()]
            if len(values) == 0:
                row.update({
                    f"{metric}_mean": math.nan, f"{metric}_std": math.nan,
                    f"{metric}_min": math.nan, f"{metric}_max": math.nan,
                    f"{metric}_median": math.nan,
                    f"{metric}_p05": math.nan, f"{metric}_p95": math.nan,
                    f"{metric}_ci_95_lower": math.nan, f"{metric}_ci_95_upper": math.nan,
                    f"{metric}_n": 0,
                })
                continue

            n = int(len(values))
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1)) if n > 1 else math.nan
            ci_lower, ci_upper = compute_confidence_interval(mean_value, std_value, n)

            row.update({
                f"{metric}_mean": mean_value,
                f"{metric}_std": std_value,
                f"{metric}_min": float(np.min(values)),
                f"{metric}_max": float(np.max(values)),
                f"{metric}_median": float(np.median(values)),
                f"{metric}_p05": float(np.percentile(values, 5)),
                f"{metric}_p95": float(np.percentile(values, 95)),
                f"{metric}_ci_95_lower": ci_lower,
                f"{metric}_ci_95_upper": ci_upper,
                f"{metric}_n": n,
            })

        rows.append(row)

    return pd.DataFrame.from_records(rows).sort_values("year").reset_index(drop=True)


def compute_demographic_indicators(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived demographic indicators to the aggregated table.

    Adds:
    - growth_rate: annual population growth rate (%)
    - births_per_1000: crude birth rate
    - deaths_per_1000: crude death rate
    - natural_increase: births - deaths
    - dependency_ratio_approx: (pop - men_women_15_64) / men_women_15_64 approximation
    - sex_ratio: men per 100 women
    """
    result = stats_df.copy()

    result["growth_rate_pct"] = result["population_mean"].pct_change() * 100.0
    result["births_per_1000"] = np.where(
        result["population_mean"] > 0,
        (result["births_mean"] / result["population_mean"]) * 1000.0,
        0.0,
    )
    result["deaths_per_1000"] = np.where(
        result["population_mean"] > 0,
        (result["deaths_mean"] / result["population_mean"]) * 1000.0,
        0.0,
    )
    result["natural_increase"] = result["births_mean"] - result["deaths_mean"]
    result["sex_ratio"] = np.where(
        result["women_mean"] > 0,
        (result["men_mean"] / result["women_mean"]) * 100.0,
        0.0,
    )
    result["couple_rate_pct"] = np.where(
        result["population_mean"] > 0,
        (result["couples_mean"] / result["population_mean"]) * 100.0,
        0.0,
    )

    return result


def print_summary(stats_df: pd.DataFrame) -> None:
    """Print a short summary of the statistical analysis."""
    if stats_df.empty:
        print("No statistics available.")
        return

    final = stats_df.iloc[-1]
    print("=" * 70)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Simulation years: {len(stats_df)}")
    print(f"  Successful runs per year: {int(final.get('successful_runs', 0))}")
    print()
    print(f"  Final year population (mean): {final['population_mean']:.1f}")
    print(f"  95% CI: [{final['population_ci_95_lower']:.1f}, {final['population_ci_95_upper']:.1f}]")
    print(f"  Range: [{final['population_min']:.0f}, {final['population_max']:.0f}]")
    print()
    print(f"  Final births/year (mean): {final['births_mean']:.1f}")
    print(f"  Final deaths/year (mean): {final['deaths_mean']:.1f}")
    print(f"  Natural increase: {final['births_mean'] - final['deaths_mean']:.1f}")
    print()
    print(f"  Sex ratio (M per 100 F): {final['men_mean'] / max(final['women_mean'], 1) * 100:.1f}")
    print(f"  Active couples (mean): {final['couples_mean']:.1f}")
    print(f"  Breakups/year (mean): {final['breakups_mean']:.1f}")
    print("=" * 70)


def export_statistics(df: pd.DataFrame, filename: str | Path = DEFAULT_OUTPUT_FILE) -> Path:
    """Save the statistics DataFrame to CSV."""
    output_path = Path(filename)
    df.to_csv(output_path, index=False)
    logging.info("Statistics exported to %s", output_path)
    return output_path


def main() -> int:
    """Run the complete simulation and export flow."""
    results = run_multiple_simulations(num_runs=DEFAULT_NUM_RUNS, seed=42)
    statistics_df = aggregate_statistics(results, target_metrics=DEFAULT_METRICS)

    if statistics_df.empty:
        logging.warning("No statistics generated because all runs failed.")
        return 1

    export_statistics(statistics_df, DEFAULT_OUTPUT_FILE)
    print_summary(statistics_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
