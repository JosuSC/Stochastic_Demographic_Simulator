"""
Adaptador limpio y desacoplado para conectar el motor de simulación real
con el módulo de análisis estadístico.

Responsabilidad única: ejecutar el simulador y transformar su salida
al formato estándar esperado por statistical_analysis.py.
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
    """Registro anual de métricas de simulación."""

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


def _count_couples(simulator: Simulator) -> int:
    """Cuenta las parejas activas en la población actual."""
    couples_set: set[tuple[int, int]] = set()
    for person in simulator.get_alive_population():
        if person.partner_id is not None:
            first_id = person.id
            second_id = person.partner_id
            pair: tuple[int, int] = (min(first_id, second_id), max(first_id, second_id))
            couples_set.add(pair)
    return len(couples_set)


def _get_sex_distribution(simulator: Simulator) -> tuple[int, int]:
    """Devuelve (hombres, mujeres) vivos."""
    alive = simulator.get_alive_population()
    men = sum(1 for p in alive if p.sex == "M")
    women = sum(1 for p in alive if p.sex == "F")
    return men, women


def run_single_simulation_complete(
    seed: Optional[int] = None,
) -> Sequence[Mapping[str, int | float]]:
    """
    Ejecuta una corrida completa de 100 años del simulador demografico.

    Devuelve una secuencia de 100 diccionarios con métricas anuales.
    Cada diccionario contiene: year, population, men, women, births, deaths, couples, breakups.

    Args:
        seed: Semilla para reproducibilidad. Si es None, usa estado aleatorio no determinista.

    Returns:
        Lista de 100 registros anuales (SimulationRecord).

    Raises:
        RuntimeError: Si el simulador falla durante la ejecución.
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
            # Capturar estado antes del año
            births_before = int(sim.total_births)
            deaths_before = int(sim.total_deaths)
            couples_before = _count_couples(sim)

            # Ejecutar el año (12 meses)
            sim.current_year_logs.clear()
            sim.run(YEARS_PER_STEP)

            # Capturar estado después del año
            alive = sim.get_alive_population()
            population = len(alive)
            men, women = _get_sex_distribution(sim)

            births_year = int(sim.total_births) - births_before
            deaths_year = int(sim.total_deaths) - deaths_before
            couples_after = _count_couples(sim)
            breakups_year = max(0, couples_before - couples_after)

            # Crear registro anual
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
