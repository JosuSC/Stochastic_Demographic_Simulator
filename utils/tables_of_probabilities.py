from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
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
    """Table of age or difference ranges with probabilities.

    It also provides a few DES-friendly probability conversions:
    - Annual to monthly: P(monthly) = 1 - (1 - P(annual))^(1/12)
    - Period from annual: P(dt) = 1 - (1 - P(annual))^dt
    - Period from monthly: P(dt) = 1 - (1 - P(monthly))^(dt*12)
    """
    ranges: List[AgeProbabilityRange] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ranges = sorted(self.ranges, key=lambda r: (r.min_age, r.max_age))

    def get_annual_probability(self, key_age: float) -> float:
        """Return the annual probability for a given age or difference."""
        for r in self.ranges:
            if r.contains(key_age):
                return r.probability
        return 0.0

    def monthly_probability(self, key_age: float) -> float:
        annual = self.get_annual_probability(key_age)
        return self.annual_to_monthly(annual)

    def period_probability(self, key_age: float, delta_years: float) -> float:
        """Compute the probability of an event during delta_years from an annual rate."""
        annual = self.get_annual_probability(key_age)
        return self.period_probability_from_annual(annual, delta_years)

    @staticmethod
    def annual_to_monthly(annual_prob: float) -> float:
        """Convert an annual probability into a monthly one."""
        if annual_prob <= 0.0:
            return 0.0
        if annual_prob >= 1.0:
            return 1.0
        return 1.0 - math.pow(1.0 - annual_prob, 1.0 / 12.0)

    @staticmethod
    def period_probability_from_annual(annual_prob: float, delta_years: float) -> float:
        """Compute the probability of an event over delta_years from an annual rate.

        Formula: P(dt) = 1 - (1 - P(annual))^dt

        This is the standard way to accumulate probability over time: the
        chance of happening at least once is one minus the chance of never
        happening.
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
        """Compute the probability of an event over delta_years from a monthly rate.

        Formula: P(dt) = 1 - (1 - P(monthly))^(dt*12)

        The event is checked once per month with probability monthly_prob.
        Over delta_years, that means delta_years * 12 checks.
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
    # DEATH PROBABILITIES (Table 1 from the problem statement)
    # ============================================================================
    # The original data gives block probabilities, so we convert them to annual
    # rates before using them in the simulator.
    # ============================================================================

    death_male = ProbabilityTable([
        AgeProbabilityRange(0, 12, 1.0 - math.pow(1.0 - 0.25, 1.0 / 12.0)),      # 25% in 12 years
        AgeProbabilityRange(12.000001, 45, 1.0 - math.pow(1.0 - 0.10, 1.0 / 33.0)),  # 10% in 33 years
        AgeProbabilityRange(45.000001, 76, 1.0 - math.pow(1.0 - 0.30, 1.0 / 31.0)),  # 30% in 31 years
        AgeProbabilityRange(76.000001, 200, 1.0 - math.pow(1.0 - 0.70, 1.0 / 49.0)), # 70% in 49 years
    ])

    death_female = ProbabilityTable([
        AgeProbabilityRange(0, 12, 1.0 - math.pow(1.0 - 0.25, 1.0 / 12.0)),      # 25% in 12 years
        AgeProbabilityRange(12.000001, 45, 1.0 - math.pow(1.0 - 0.15, 1.0 / 33.0)),  # 15% in 33 years
        AgeProbabilityRange(45.000001, 76, 1.0 - math.pow(1.0 - 0.35, 1.0 / 31.0)),  # 35% in 31 years
        AgeProbabilityRange(76.000001, 200, 1.0 - math.pow(1.0 - 0.65, 1.0 / 49.0)), # 65% in 49 years
    ])

    # ============================================================================
    # PREGNANCY PROBABILITIES (Table 2)
    # Annual chance of pregnancy by age.
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
    # PARTNER DESIRE (Table 4)
    # Annual chance of wanting a partner by age.
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
    # COUPLE MATCH (Table 5)
    # Chance of forming a couple based on age difference.
    # Treated as the probability of a single attempt.
    # ============================================================================
    couple_match = ProbabilityTable([
        AgeProbabilityRange(0.0, 5.0, 0.45),
        AgeProbabilityRange(5.0001, 10.0, 0.40),
        AgeProbabilityRange(10.0001, 15.0, 0.35),
        AgeProbabilityRange(15.0001, 20.0, 0.25),
        AgeProbabilityRange(20.0001, 200.0, 0.15),
    ])

    # ============================================================================
    # BREAKUP
    # Constant annual breakup probability.
    # ============================================================================
    breakup = ProbabilityTable([AgeProbabilityRange(0.0, 200.0, 0.20)])

    # ============================================================================
    # GRIEF TIME (Table 6)
    # Average time alone after a breakup or widowhood.
    # ============================================================================
    grief_time = GriefTimeTable([
        GriefTimeRange(0.0, 15.0, 3.0),       # 3 months
        GriefTimeRange(15.0001, 21.0, 6.0),    # 6 months
        GriefTimeRange(21.0001, 35.0, 6.0),    # 6 months
        GriefTimeRange(35.0001, 45.0, 12.0),   # 1 year
        GriefTimeRange(45.0001, 60.0, 24.0),   # 2 years
        GriefTimeRange(60.0001, 200.0, 48.0),  # 4 years
    ])

    # ============================================================================
    # BABIES DISTRIBUTION (Table 7)
    # Babies per pregnancy.
    # ============================================================================
    babies_distribution = DiscreteDistribution([
        DistributionEntry(1, 0.70),
        DistributionEntry(2, 0.18),
        DistributionEntry(3, 0.08),
        DistributionEntry(4, 0.04),
        DistributionEntry(5, 0.02),
    ])

    # ============================================================================
    # DESIRED CHILDREN (Table 3)
    # Desired children per person.
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


# One shared registry for the whole project.
tables: ProbabilityTablesRegistry = _make_default_registry()

__all__ = [
    "AgeProbabilityRange",
    "ProbabilityTable",
    "GriefTimeTable",
    "DiscreteDistribution",
    "ProbabilityTablesRegistry",
    "tables",
]
