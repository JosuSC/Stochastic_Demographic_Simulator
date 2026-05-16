from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable
import bisect
import random
import math


@dataclass(frozen=True)
class AgeProbabilityRange:
    min_age: float
    max_age: float
    probability: float

    def contains(self, age: float) -> bool:
        return self.min_age <= age <= self.max_age


@dataclass
class ProbabilityTable:
    """Object-oriented table storing age/difference ranges with probabilities.

    - Stores ranges as objects (not tuples)
    - Provides type-safe lookups and DES-aware conversions
    """
    ranges: List[AgeProbabilityRange] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ranges = sorted(self.ranges, key=lambda r: (r.min_age, r.max_age))

    def get_annual_probability(self, key_age: float) -> float:
        for r in self.ranges:
            if r.contains(key_age):
                return r.probability
        return 0.0

    def monthly_probability(self, key_age: float) -> float:
        annual = self.get_annual_probability(key_age)
        return self.annual_to_monthly(annual)

    def period_probability(self, key_age: float, delta_years: float) -> float:
        annual = self.get_annual_probability(key_age)
        return self.period_probability_from_annual(annual, delta_years)

    @staticmethod
    def annual_to_monthly(annual_prob: float) -> float:
        """Convierte probabilidad anual a mensual: P(mes) = 1 - (1 - P(anual))^(1/12)"""
        if annual_prob <= 0.0:
            return 0.0
        if annual_prob >= 1.0:
            return 1.0
        return 1.0 - math.pow(1.0 - annual_prob, 1.0 / 12.0)

    @staticmethod
    def period_probability_from_annual(annual_prob: float, delta_years: float) -> float:
        """Calcula la probabilidad de que un evento ocurra en un período de delta_years,
        dada una probabilidad anual.

        Formula: P(período) = 1 - (1 - P(anual))^(delta_years)

        Esta es la forma correcta de acumular probabilidad en un DES:
        si la probabilidad anual es p, la probabilidad de que el evento
        ocurra AL MENOS UNA VEZ en delta_years es 1-(1-p)^delta.
        """
        if annual_prob <= 0.0:
            return 0.0
        if annual_prob >= 1.0:
            return 1.0
        if delta_years <= 0.0:
            return 0.0
        return 1.0 - math.pow(1.0 - annual_prob, delta_years)

    @staticmethod
    def period_probability_from_monthly(monthly_prob: float, delta_years: float) -> float:
        """Calcula la probabilidad de que un evento ocurra en un período de delta_years,
        dada una probabilidad mensual.

        Formula: P(período) = 1 - (1 - P(mensual))^(delta_years * 12)

        El evento se verifica mensualmente con probabilidad monthly_prob.
        En delta_years hay delta_years*12 meses, y la probabilidad de que
        NO ocurra en ninguno es (1-p)^(n_meses), por lo que la probabilidad
        de que ocurra AL MENOS UNA VEZ es 1-(1-p)^(n_meses).
        """
        if monthly_prob <= 0.0:
            return 0.0
        if monthly_prob >= 1.0:
            return 1.0
        if delta_years <= 0.0:
            return 0.0
        delta_months = delta_years * 12.0
        return 1.0 - math.pow(1.0 - monthly_prob, delta_months)


@dataclass
class GriefTimeRange:
    min_age: float
    max_age: float
    mean_months: float

    def contains(self, age: float) -> bool:
        return self.min_age <= age <= self.max_age


@dataclass
class GriefTimeTable:
    ranges: List[GriefTimeRange] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ranges = sorted(self.ranges, key=lambda r: (r.min_age, r.max_age))

    def get_mean_months(self, age: float) -> float:
        for r in self.ranges:
            if r.contains(age):
                return r.mean_months
        return 12.0


@dataclass(frozen=True)
class DistributionEntry:
    value: int
    probability: float


@dataclass
class DiscreteDistribution:
    entries: List[DistributionEntry] = field(default_factory=list)
    _cum: List[float] = field(init=False, default_factory=list)
    _values: List[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        # normalize and build cumulative
        total = sum(e.probability for e in self.entries)
        if total <= 0:
            raise ValueError("Distribution must have positive total probability")
        cum = 0.0
        self._cum = []
        self._values = []
        for e in self.entries:
            prob = e.probability / total
            cum += prob
            self._cum.append(cum)
            self._values.append(e.value)

    def sample(self) -> int:
        r = random.random()
        idx = bisect.bisect_left(self._cum, r)
        if idx < 0:
            idx = 0
        if idx >= len(self._values):
            idx = len(self._values) - 1
        return self._values[idx]


@dataclass
class ProbabilityTablesRegistry:
    death_male: ProbabilityTable
    death_female: ProbabilityTable
    pregnancy: ProbabilityTable
    partner_desire: ProbabilityTable
    couple_match: ProbabilityTable
    breakup: ProbabilityTable
    grief_time: GriefTimeTable
    babies_distribution: DiscreteDistribution
    desired_children_distribution: DiscreteDistribution


def _make_default_registry() -> ProbabilityTablesRegistry:
    # ============================================================================
    # DEATH PROBABILITIES (Tabla 1 del problema)
    # ============================================================================
    # La tabla del problema da probabilidades de morir DENTRO de cada rango de edad.
    # Ejemplo: Hombre 0-12 tiene probabilidad 0.25 de morir en esos 12 años.
    #
    # Para convertir a probabilidad ANUAL usamos:
    #   P(rango) = 1 - (1 - P_anual)^(anios_del_rango)
    #   => P_anual = 1 - (1 - P_rango)^(1/anios_del_rango)
    #
    # Verificacion:
    #   Hombre 0-12: P_anual = 1 - (1-0.25)^(1/12) = 0.02394
    #   => P(12 años) = 1 - (1-0.02394)^12 = 1 - 0.75 = 0.25 ✓
    # ============================================================================

    # Hombre
    death_male = ProbabilityTable([
        AgeProbabilityRange(0, 12, 1.0 - math.pow(1.0 - 0.25, 1.0 / 12.0)),     # 0.25 en 12 años
        AgeProbabilityRange(12.000001, 45, 1.0 - math.pow(1.0 - 0.10, 1.0 / 33.0)),  # 0.10 en 33 años
        AgeProbabilityRange(45.000001, 76, 1.0 - math.pow(1.0 - 0.30, 1.0 / 31.0)),  # 0.30 en 31 años
        AgeProbabilityRange(76.000001, 200, 1.0 - math.pow(1.0 - 0.70, 1.0 / 49.0)), # 0.70 en 49 años
    ])

    # Mujer
    death_female = ProbabilityTable([
        AgeProbabilityRange(0, 12, 1.0 - math.pow(1.0 - 0.25, 1.0 / 12.0)),     # 0.25 en 12 años
        AgeProbabilityRange(12.000001, 45, 1.0 - math.pow(1.0 - 0.15, 1.0 / 33.0)),  # 0.15 en 33 años
        AgeProbabilityRange(45.000001, 76, 1.0 - math.pow(1.0 - 0.35, 1.0 / 31.0)),  # 0.35 en 31 años
        AgeProbabilityRange(76.000001, 200, 1.0 - math.pow(1.0 - 0.65, 1.0 / 49.0)), # 0.65 en 49 años
    ])

    # ============================================================================
    # PREGNANCY PROBABILITIES (Tabla 2 del problema)
    # ============================================================================
    # Probabilidad ANUAL de que una mujer quede embarazada según su edad.
    # ============================================================================
    pregnancy = ProbabilityTable([
        AgeProbabilityRange(0, 11.9999, 0.0),
        AgeProbabilityRange(12.0, 15.0, 0.20),
        AgeProbabilityRange(15.0001, 21.0, 0.45),
        AgeProbabilityRange(21.0001, 35.0, 0.80),
        AgeProbabilityRange(35.0001, 45.0, 0.40),
        AgeProbabilityRange(45.0001, 60.0, 0.20),
        AgeProbabilityRange(60.0001, 200.0, 0.05),
    ])

    # ============================================================================
    # PARTNER DESIRE (Tabla 4 del problema)
    # ============================================================================
    # Probabilidad ANUAL de querer pareja según la edad.
    # ============================================================================
    partner_desire = ProbabilityTable([
        AgeProbabilityRange(0, 11.9999, 0.0),
        AgeProbabilityRange(12.0, 15.0, 0.60),
        AgeProbabilityRange(15.0001, 21.0, 0.65),
        AgeProbabilityRange(21.0001, 35.0, 0.80),
        AgeProbabilityRange(35.0001, 45.0, 0.60),
        AgeProbabilityRange(45.0001, 60.0, 0.50),
        AgeProbabilityRange(60.0001, 200.0, 0.20),
    ])

    # ============================================================================
    # COUPLE MATCH (Tabla 5 del problema)
    # ============================================================================
    # Probabilidad de formar pareja según la diferencia de edad.
    # Se interpreta como probabilidad por intento de formación.
    # ============================================================================
    couple_match = ProbabilityTable([
        AgeProbabilityRange(0.0, 5.0, 0.45),
        AgeProbabilityRange(5.0001, 10.0, 0.40),
        AgeProbabilityRange(10.0001, 15.0, 0.35),
        AgeProbabilityRange(15.0001, 20.0, 0.25),
        AgeProbabilityRange(20.0001, 200.0, 0.15),
    ])

    # ============================================================================
    # BREAKUP (enunciado del problema)
    # ============================================================================
    # Probabilidad ANUAL de ruptura = 0.20 (constante).
    # ============================================================================
    breakup = ProbabilityTable([AgeProbabilityRange(0.0, 200.0, 0.20)])

    # ============================================================================
    # GRIEF TIME (Tabla 6 del problema)
    # ============================================================================
    # Tiempo de duelo en soledad: distribución exponencial con media dada.
    # Los valores de la tabla representan la MEDIA del tiempo de duelo.
    # ============================================================================
    grief_time = GriefTimeTable([
        GriefTimeRange(0.0, 15.0, 3.0),       # 3 meses
        GriefTimeRange(15.0001, 21.0, 6.0),    # 6 meses
        GriefTimeRange(21.0001, 35.0, 6.0),    # 6 meses
        GriefTimeRange(35.0001, 45.0, 12.0),   # 1 año = 12 meses
        GriefTimeRange(45.0001, 60.0, 24.0),   # 2 años = 24 meses
        GriefTimeRange(60.0001, 200.0, 48.0),  # 4 años = 48 meses
    ])

    # ============================================================================
    # BABIES DISTRIBUTION (Tabla 7 del problema)
    # ============================================================================
    # Distribución del número de bebés por embarazo.
    # ============================================================================
    babies_distribution = DiscreteDistribution([
        DistributionEntry(1, 0.70),
        DistributionEntry(2, 0.18),
        DistributionEntry(3, 0.08),
        DistributionEntry(4, 0.04),
        DistributionEntry(5, 0.02),
    ])

    # ============================================================================
    # DESIRED CHILDREN (Tabla 3 del problema)
    # ============================================================================
    # Distribución del número deseado de hijos por persona.
    # ============================================================================
    desired_children_distribution = DiscreteDistribution([
        DistributionEntry(1, 0.60),
        DistributionEntry(2, 0.75),
        DistributionEntry(3, 0.35),
        DistributionEntry(4, 0.20),
        DistributionEntry(5, 0.10),
        DistributionEntry(6, 0.05),
    ])

    return ProbabilityTablesRegistry(
        death_male=death_male,
        death_female=death_female,
        pregnancy=pregnancy,
        partner_desire=partner_desire,
        couple_match=couple_match,
        breakup=breakup,
        grief_time=grief_time,
        babies_distribution=babies_distribution,
        desired_children_distribution=desired_children_distribution,
    )


# single shared registry instance for simple import
tables: ProbabilityTablesRegistry = _make_default_registry()


# Exported names
__all__ = [
    "AgeProbabilityRange",
    "ProbabilityTable",
    "GriefTimeTable",
    "DiscreteDistribution",
    "ProbabilityTablesRegistry",
    "tables",
]
