"""
Adapter that bridges the simulator and the statistical analysis module.

Its job is simple: run the simulator and reshape the output into the format
expected by statistical_analysis.py.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence, Mapping

from engine.simulator import Simulator


SIMULATION_YEARS = 100
YEARS_PER_STEP = 1.0
INITIAL_POPULATION_FEMALE = 500
INITIAL_POPULATION_MALE = 500


class SimulationRecord(dict[str, int | float]):
    """Yearly metrics for one simulation run."""

    def __init__(
        self,
        year: int,
        population: int,
        men: int,
        women: int,
        births: int,
        deaths: int,
        couples: int,
        breakups: int,
    ) -> None:
        super().__init__(
            year=year,
            population=population,
            men=men,
            women=women,
            births=births,
            deaths=deaths,
            couples=couples,
            breakups=breakups,
        )


def run_single_simulation_complete(
    seed: Optional[int] = None,
) -> Sequence[Mapping[str, int | float]]:
    """
    Run one full 100-year simulation and return the yearly metrics.

    The result contains one dictionary per year with:
    year, population, men, women, births, deaths, couples, breakups.

    Args:
        seed: Seed for reproducibility. None keeps it random.

    Raises:
        RuntimeError: If the simulator fails while running.
    """
    if seed is not None:
        random.seed(seed)

    try:
        sim = Simulator(
            initial_females=INITIAL_POPULATION_FEMALE,
            initial_males=INITIAL_POPULATION_MALE,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize simulation: {exc}") from exc

    records: list[Mapping[str, int | float]] = []

    for year_index in range(SIMULATION_YEARS):
        try:
            births_before = int(sim.total_births)
            deaths_before = int(sim.total_deaths)
            breakups_before = int(sim.total_breakups)
            couples_before = sim.count_couples()

            sim.current_year_logs.clear()
            sim.run(YEARS_PER_STEP)

            alive = sim.population_alive
            population = len(alive)
            men = sum(1 for p in alive if p.sex == "M")
            women = population - men

            births_year = int(sim.total_births) - births_before
            deaths_year = int(sim.total_deaths) - deaths_before
            breakups_year = int(sim.total_breakups) - breakups_before
            couples_after = sim.count_couples()

            record = SimulationRecord(
                year=year_index,
                population=population,
                men=men,
                women=women,
                births=births_year,
                deaths=deaths_year,
                couples=couples_after,
                breakups=breakups_year,
            )
            records.append(record)

        except Exception as exc:
            raise RuntimeError(
                f"Simulation failed at year {year_index}: {exc}"
            ) from exc

    if len(records) != SIMULATION_YEARS:
        raise RuntimeError(
            f"Expected {SIMULATION_YEARS} annual records, got {len(records)}"
        )

    return records
