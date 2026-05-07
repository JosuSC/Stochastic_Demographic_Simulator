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
DEFAULT_NUM_RUNS = 100
DEFAULT_OUTPUT_FILE = Path("simulation_statistics.csv")
DEFAULT_METRICS = ("population", "births", "deaths", "couples")


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
    """Resultado de una corrida individual de la simulación."""

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
    """Registra el motor real que ejecuta una corrida de simulación."""

    _runner_registry.runner = runner


def run_single_simulation(seed: Optional[int] = None) -> list[YearlySimulationRecord]:
    """Ejecuta una corrida individual mediante el runner registrado.

    Este módulo no inventa el simulador. Debes registrar tu motor real con
    register_simulation_runner() y devolver una secuencia de 100 diccionarios,
    uno por año.
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
    """Normaliza y valida el formato esperado de una corrida."""

    if len(records) != YEARS_PER_SIMULATION:
        raise ValueError(
            f"Each simulation must return {YEARS_PER_SIMULATION} yearly records. Got {len(records)}."
        )

    normalized: list[YearlySimulationRecord] = []
    required_keys = (
        "year",
        "population",
        "men",
        "women",
        "births",
        "deaths",
        "couples",
        "breakups",
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
    """Genera una semilla reproducible por corrida."""

    if seed is None:
        return None

    generator = np.random.default_rng(seed)
    derived_seeds = generator.integers(0, np.iinfo(np.int32).max, size=run_index + 1)
    return int(derived_seeds[-1])


def run_multiple_simulations(
    num_runs: int = DEFAULT_NUM_RUNS,
    seed: Optional[int] = None,
) -> list[SimulationRunResult]:
    """Ejecuta múltiples corridas independientes y conserva errores sin abortar."""

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
            NotImplementedError,
            ValueError,
            RuntimeError,
            TypeError,
            KeyError,
            IndexError,
            OSError,
            AssertionError,
            ArithmeticError,
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
    """Calcula el intervalo de confianza al 95% para una media muestral."""

    if n <= 1 or not math.isfinite(mean) or not math.isfinite(std):
        return mean, mean

    margin_of_error = 1.96 * (std / math.sqrt(n))
    return mean - margin_of_error, mean + margin_of_error


def aggregate_statistics(
    results: Sequence[SimulationRunResult],
    target_metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Agrega estadísticas anuales sobre las corridas exitosas."""

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
                row.update(
                    {
                        f"{metric}_mean": math.nan,
                        f"{metric}_std": math.nan,
                        f"{metric}_min": math.nan,
                        f"{metric}_max": math.nan,
                        f"{metric}_median": math.nan,
                        f"{metric}_p05": math.nan,
                        f"{metric}_p95": math.nan,
                        f"{metric}_ci_95_lower": math.nan,
                        f"{metric}_ci_95_upper": math.nan,
                        f"{metric}_n": 0,
                    }
                )
                continue

            n = int(len(values))
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1)) if n > 1 else math.nan
            ci_lower, ci_upper = compute_confidence_interval(mean_value, std_value, n)

            row.update(
                {
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
                }
            )

        rows.append(row)

    return pd.DataFrame.from_records(rows).sort_values("year").reset_index(drop=True)


def export_statistics(df: pd.DataFrame, filename: str | Path = DEFAULT_OUTPUT_FILE) -> Path:
    """Exporta el DataFrame estadístico a CSV y devuelve la ruta generada."""

    output_path = Path(filename)
    df.to_csv(output_path, index=False)
    logging.info("Statistics exported to %s", output_path)
    return output_path


def main() -> int:
    """Ejecuta el flujo completo de simulación y exportación."""

    results = run_multiple_simulations(num_runs=DEFAULT_NUM_RUNS, seed=42)
    statistics_df = aggregate_statistics(results, target_metrics=DEFAULT_METRICS)

    if statistics_df.empty:
        logging.warning("No statistics were generated because all runs failed.")
        return 1

    export_statistics(statistics_df, DEFAULT_OUTPUT_FILE)
    logging.info("Statistical analysis completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
