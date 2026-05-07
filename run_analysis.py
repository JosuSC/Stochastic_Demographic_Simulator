"""
Script de integración: conecta el adaptador con el análisis estadístico.

Flujo completo:
1. Registra el adaptador con statistical_analysis
2. Ejecuta N corridas independientes
3. Agrega estadísticas anuales
4. Exporta a CSV
"""

import logging
from pathlib import Path

from simulation_adapter import run_single_simulation_complete
from statistical_analysis import (
    register_simulation_runner,
    run_multiple_simulations,
    aggregate_statistics,
    export_statistics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> int:
    """Ejecuta el flujo completo de simulación y análisis."""

    # 1. Registrar el adaptador como el ejecutor de simulaciones
    logging.info("Registrando adaptador de simulación...")
    register_simulation_runner(run_single_simulation_complete)

    # 2. Ejecutar 100 corridas independientes con semilla reproducible
    logging.info("Ejecutando 100 corridas de simulación...")
    results = run_multiple_simulations(num_runs=100, seed=42)

    # 3. Agregar estadísticas anuales
    logging.info("Agregando estadísticas anuales...")
    statistics_df = aggregate_statistics(
        results,
        target_metrics=("population", "births", "deaths", "couples", "men", "women", "breakups"),
    )

    if statistics_df.empty:
        logging.error("No se generaron estadísticas (todas las corridas fallaron)")
        return 1

    # 4. Exportar a CSV
    output_file = Path("simulation_statistics.csv")
    logging.info("Exportando resultados a %s", output_file)
    export_statistics(statistics_df, output_file)

    # 5. Resumen final
    logging.info("=== RESUMEN FINAL ===")
    logging.info("Años simulados: %d", len(statistics_df))
    logging.info("Columnas en el reporte: %s", ", ".join(statistics_df.columns.tolist()))
    logging.info("Primeras 5 filas:")
    logging.info("\n%s", statistics_df.head().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
