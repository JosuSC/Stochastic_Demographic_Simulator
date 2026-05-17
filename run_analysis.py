"""
Glue script that connects the simulator adapter with the statistics module.

Pipeline:
1. Register the adapter in statistical_analysis
2. Run N independent simulations
3. Aggregate yearly statistics
4. Export raw run data if needed
5. Write the aggregated CSV
6. Print a short summary
"""

import logging
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from simulation_adapter import run_single_simulation_complete
from statistical_analysis import (
    SimulationRunResult,
    register_simulation_runner,
    run_multiple_simulations,
    aggregate_statistics,
    compute_demographic_indicators,
    print_summary,
    export_statistics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def export_raw_runs(results: Sequence[SimulationRunResult], output_file: Path) -> None:
    """Write the yearly detail for each run so it can be plotted later."""

    rows: list[dict[str, object]] = []
    for result in results:
        if not result.succeeded or result.records is None:
            continue
        for record in result.records:
            rows.append({
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
            })

    if not rows:
        logging.warning("No raw run data generated, skipping %s", output_file)
        return

    pd.DataFrame.from_records(rows).to_csv(output_file, index=False)
    logging.info("Raw runs exported to %s", output_file)


def main() -> int:
    """Run the full simulation and analysis flow."""

    # 1. Plug the adapter into the analysis layer.
    logging.info("Registering simulation adapter...")
    register_simulation_runner(run_single_simulation_complete)

    # 2. Launch the independent runs.
    num_runs = 30
    logging.info("Running %d simulation iterations...", num_runs)
    start_time = time.time()
    results = run_multiple_simulations(num_runs=num_runs, seed=42)
    elapsed = time.time() - start_time
    logging.info("All runs completed in %.1f seconds (%.1f s/run average)",
                 elapsed, elapsed / max(num_runs, 1))

    # 3. Combine the yearly statistics.
    logging.info("Aggregating annual statistics...")
    statistics_df = aggregate_statistics(
        results,
        target_metrics=("population", "births", "deaths", "couples", "men", "women", "breakups"),
    )

    if statistics_df.empty:
        logging.error("No statistics generated (all runs failed)")
        return 1

    # 4. Add the derived demographic indicators.
    enhanced_df = compute_demographic_indicators(statistics_df)

    # 5. Save the raw run data.
    raw_output_file = Path("simulation_runs.csv")
    export_raw_runs(results, raw_output_file)

    # 6. Save the aggregated statistics.
    output_file = Path("simulation_statistics.csv")
    logging.info("Exporting results to %s", output_file)
    export_statistics(enhanced_df, output_file)

    # 7. Print the final summary.
    print_summary(enhanced_df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
