"""
Script de integración: conecta el adaptador con el análisis estadístico.

Flujo completo:
1. Registra el adaptador con statistical_analysis
2. Ejecuta N corridas independientes
3. Agrega estadísticas anuales
4. Exporta las corridas crudas opcionalmente
5. Exporta a CSV
"""

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from simulation_adapter import run_single_simulation_complete
from statistical_analysis import (
    SimulationRunResult,
    register_simulation_runner,
    run_multiple_simulations,
    aggregate_statistics,
    export_statistics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def export_raw_runs(results: Sequence[SimulationRunResult], output_file: Path) -> None:
    """Guarda el detalle anual de cada corrida para visualizaciones de distribución."""

    rows: list[dict[str, object]] = []
    for result in results:
        if not result.succeeded or result.records is None:
            continue
        for record in result.records:
            rows.append(
                {
                    "run_id": result.run_id,
                    "seed": result.seed,
                    "year": record["year"],
                    "population": record["population"],
                    "men": record["men"],
                    "women": record["women"],
                    "births": record["births"],
                    "deaths": record["deaths"],
                    "couples": record["couples"],
                    "breakups": record["breakups"],
                }
            )

    if not rows:
        logging.warning("No raw run data was generated, skipping %s", output_file)
        return

    pd.DataFrame.from_records(rows).to_csv(output_file, index=False)
    logging.info("Corridas crudas exportadas a %s", output_file)


def main(
    num_runs: int = 100,
    seed: int | None = 42,
    raw_output: Path = Path("simulation_runs.csv"),
    stats_output: Path = Path("simulation_statistics.csv"),
) -> int:
    """Ejecuta el flujo completo de simulación y análisis.

    Args:
        num_runs: Número de corridas independientes a ejecutar.
        seed: Semilla base para reproducibilidad.
        raw_output: Ruta para CSV con corridas crudas.
        stats_output: Ruta para CSV con estadísticas agregadas.
    """

    # 1. Registrar el adaptador como el ejecutor de simulaciones
    logging.info("Registrando adaptador de simulación...")
    register_simulation_runner(run_single_simulation_complete)

    # 2. Ejecutar corridas independientes con semilla reproducible
    logging.info("Ejecutando %d corridas de simulación...", num_runs)
    results = run_multiple_simulations(num_runs=num_runs, seed=seed)

    # 3. Agregar estadísticas anuales
    logging.info("Agregando estadísticas anuales...")
    statistics_df = aggregate_statistics(
        results,
        target_metrics=("population", "births", "deaths", "couples", "men", "women", "breakups"),
    )

    if statistics_df.empty:
        logging.error("No se generaron estadísticas (todas las corridas fallaron)")
        return 1

    # 3b. Exportar detalle crudo para histogramas y análisis de distribución
    export_raw_runs(results, Path(raw_output))

    # 4. Exportar a CSV
    logging.info("Exportando resultados a %s", stats_output)
    export_statistics(statistics_df, Path(stats_output))

    # 5. Resumen final
    logging.info("=== RESUMEN FINAL ===")
    logging.info("Años simulados: %d", len(statistics_df))
    logging.info("Columnas en el reporte: %s", ", ".join(statistics_df.columns.tolist()))
    logging.info("Primeras 5 filas:")
    logging.info("\n%s", statistics_df.head().to_string())

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run simulation analysis and export statistics.")
    parser.add_argument("--runs", type=int, default=100, help="Number of independent runs")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (optional)")
    parser.add_argument("--raw-output", type=Path, default=Path("simulation_runs.csv"), help="CSV for raw runs")
    parser.add_argument("--stats-output", type=Path, default=Path("simulation_statistics.csv"), help="CSV for aggregated statistics")

    args = parser.parse_args()
    raise SystemExit(main(num_runs=args.runs, seed=args.seed, raw_output=args.raw_output, stats_output=args.stats_output))
