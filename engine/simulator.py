"""
DES engine for the demographic population simulation.

Main optimizations:
1. Births are handled only by BIRTH_EVENT, so pregnancies are not scanned on
    every event.
2. The dispatch table is built once in __init__.
3. Events are stored as tuples to keep comparisons cheap.
4. Grief is tracked with an absolute end time.
5. Age checks stay inline in the hot paths.
6. Probability lookups use bisect instead of a linear scan.
7. Partner and child events for children under 12 are delayed until age 12.
8. Alive people are tracked with sets for fast membership checks.
"""
from __future__ import annotations

import heapq
import math
import random
import bisect
from typing import Dict, List, Optional, Set, Tuple

from models.person import Person
from utils.probability import get_uniform, evaluate_probability, get_exponential
from utils.tables_of_probabilities import tables, ProbabilityTable


# ---------------------------------------------------------------------------
# Event representation: a lightweight tuple keeps this fast.
# (time, seq, event_type_str, target_id, payload_dict)
# ---------------------------------------------------------------------------

# Event type constants.
EVT_DEATH = 0
EVT_PARTNER_DESIRE = 1
EVT_COUPLE_FORMATION = 2
EVT_CHILD_DESIRE = 3
EVT_PREGNANCY_ATTEMPT = 4
EVT_PREGNANCY_START = 5
EVT_BIRTH = 6
EVT_BREAKUP = 7

EVT_NAMES = {
    EVT_DEATH: "DEATH",
    EVT_PARTNER_DESIRE: "PARTNER_DESIRE",
    EVT_COUPLE_FORMATION: "COUPLE_FORMATION",
    EVT_CHILD_DESIRE: "CHILD_DESIRE",
    EVT_PREGNANCY_ATTEMPT: "PREGNANCY_ATTEMPT",
    EVT_PREGNANCY_START: "PREGNANCY_START",
    EVT_BIRTH: "BIRTH",
    EVT_BREAKUP: "BREAKUP",
}


class GestationRecord:
    __slots__ = ('fetus_id', 'mother_id', 'father_id', 'due_time')

    def __init__(self, fetus_id: int, mother_id: int,
                 father_id: Optional[int], due_time: float):
        self.fetus_id = fetus_id
        self.mother_id = mother_id
        self.father_id = father_id
        self.due_time = due_time


class RelationshipRecord:
    __slots__ = ('partner_a', 'partner_b', 'last_breakup_check_time',
                 'last_pregnancy_attempt_time')

    def __init__(self, partner_a: int, partner_b: int,
                 last_breakup_check_time: float,
                 last_pregnancy_attempt_time: float):
        self.partner_a = partner_a
        self.partner_b = partner_b
        self.last_breakup_check_time = last_breakup_check_time
        self.last_pregnancy_attempt_time = last_pregnancy_attempt_time


class Simulator:
    """DES engine with continuous time measured in years.

    Ages are computed lazily: Person._current_sim_time is updated before each
    event, and person.age_years is read only when needed.

    Key formula: if P_annual is the annual probability of an event, then the
    probability over dt years is:
        P(dt) = 1 - (1 - P_annual)^dt
    """

    # Event check rates, expressed as events per year.
    DEATH_CHECK_RATE = 12.0          # Once per month.
    PARTNER_DESIRE_RATE = 4.0        # About once per quarter.
    CHILD_DESIRE_RATE = 4.0          # About once per quarter.
    COUPLE_FORMATION_BASE_RATE = 24.0  # Scales with population size.
    PREGNANCY_ATTEMPT_RATE = 12.0    # Once per month per couple.
    BREAKUP_CHECK_RATE = 6.0         # Roughly every two months per couple.

    def __init__(self, initial_females: int, initial_males: int):
        self.current_time_years: float = 0.0
        self.total_births: int = 0
        self.total_deaths: int = 0
        self.total_breakups: int = 0
        self.total_couples_formed: int = 0

        self.people: Dict[int, Person] = {}
        self.alive_ids: Set[int] = set()
        self.dead_ids: Set[int] = set()
        self.male_group: Set[int] = set()
        self.female_group: Set[int] = set()
        self.gestation_group: Dict[int, GestationRecord] = {}
        self.mother_gestations: Dict[int, Set[int]] = {}
        self.relationships: Dict[Tuple[int, int], RelationshipRecord] = {}
        self.ready_males: Set[int] = set()
        self.ready_females: Set[int] = set()

        # Future event list as a simple tuple heap.
        self.fel: list = []
        self._event_seq: int = 0
        self._schedule_built: bool = False

        self.current_year_logs: List[str] = []

        # Build the dispatch table once instead of per event.
        self._dispatch = {
            EVT_DEATH: self._handle_death_event,
            EVT_PARTNER_DESIRE: self._handle_partner_desire_event,
            EVT_COUPLE_FORMATION: self._handle_couple_formation_event,
            EVT_CHILD_DESIRE: self._handle_child_desire_event,
            EVT_PREGNANCY_ATTEMPT: self._handle_pregnancy_attempt_event,
            EVT_PREGNANCY_START: self._handle_pregnancy_start_event,
            EVT_BIRTH: self._handle_birth_event,
            EVT_BREAKUP: self._handle_breakup_event,
        }

        # Pre-compute lookup tables for bisect.
        self._death_male = self._build_bisect_table(tables.death_male)
        self._death_female = self._build_bisect_table(tables.death_female)
        self._pregnancy = self._build_bisect_table(tables.pregnancy)
        self._partner_desire = self._build_bisect_table(tables.partner_desire)
        self._couple_match = self._build_bisect_table(tables.couple_match)

        # Set the shared simulation clock before creating people.
        Person.set_sim_time(0.0)

        self._initialize_population(initial_females, "F")
        self._initialize_population(initial_males, "M")

    @staticmethod
    def _build_bisect_table(ptable: ProbabilityTable):
        """Prepare sorted boundaries for O(log R) lookups."""
        if not ptable.ranges:
            return ([], [])
        boundaries = []
        probs = []
        for r in ptable.ranges:
            boundaries.append(r.max_age)
            probs.append(r.probability)
        return (boundaries, probs)

    def _fast_lookup(self, bisect_table, age: float) -> float:
        """Look up a probability with bisect in O(log R)."""
        boundaries, probs = bisect_table
        if not boundaries:
            return 0.0
        idx = bisect.bisect_left(boundaries, age)
        if idx >= len(probs):
            return 0.0
        return probs[idx]

    # --- Population queries ---

    @property
    def population_alive(self) -> List[Person]:
        """Return the alive people list when it is requested."""
        return [self.people[pid] for pid in self.alive_ids if pid in self.people]

    @property
    def population_dead(self) -> List[Person]:
        return [self.people[pid] for pid in self.dead_ids if pid in self.people]

    @property
    def current_month(self) -> float:
        return self.current_time_years * 12.0

    def _log_event(self, message: str) -> None:
        self.current_year_logs.append(message)

    def _initialize_population(self, count: int, sex: str) -> None:
        for _ in range(count):
            age_years = random.uniform(0, 100)
            person = Person(sex, age_years)
            person.desired_children = tables.desired_children_distribution.sample()
            self._add_person_to_population(person)

    def _add_person_to_population(self, person: Person) -> None:
        """Add a living person to the simulator indexes."""
        self.people[person.id] = person
        self.alive_ids.add(person.id)
        if person.sex == "M":
            self.male_group.add(person.id)
        else:
            self.female_group.add(person.id)

    def _get_grief_time_years(self, age_years: float) -> float:
        mean_months = tables.grief_time.get_mean_months(age_years)
        grief_months = get_exponential(mean_months)
        return max(1.0 / 12.0, grief_months / 12.0)

    def _wants_partner(self, person: Person) -> bool:
        if person.partner_id is not None:
            return False
        if person.is_grieving(self.current_time_years):
            return False

        age = person.age_years
        annual_prob = self._fast_lookup(self._partner_desire, age)
        if annual_prob <= 0:
            return False

        delta = self.current_time_years - person.last_partner_desire_check_time
        if delta <= 0:
            return evaluate_probability(annual_prob)

        period_prob = 1.0 - math.pow(1.0 - annual_prob, delta)
        return evaluate_probability(period_prob)

    def _match_probability(self, p1_age: float, p2_age: float) -> float:
        if (p1_age < 18) != (p2_age < 18):
            return 0.0
        diff = abs(p1_age - p2_age)
        return self._fast_lookup(self._couple_match, diff)

    def _pair_key(self, a_id: int, b_id: int) -> Tuple[int, int]:
        return (min(a_id, b_id), max(a_id, b_id))

    def _couple_formation_rate(self) -> float:
        n_ready = len(self.ready_males) + len(self.ready_females)
        if n_ready <= 0:
            return self.COUPLE_FORMATION_BASE_RATE
        return max(self.COUPLE_FORMATION_BASE_RATE, n_ready * 2.0)

    def schedule_event(self, time: float, evt_type: int,
                       target_id: int = -1,
                       payload: Optional[Dict] = None) -> None:
        """Push a new event into the future event list."""
        if time < self.current_time_years:
            time = self.current_time_years
        seq = self._event_seq
        self._event_seq += 1
        if payload is None:
            payload = {}
        # Stored as: (time, seq, evt_type, target_id, payload)
        heapq.heappush(self.fel, (time, seq, evt_type, target_id, payload))

    def build_initial_schedule(self) -> None:
        if self._schedule_built:
            return
        self._schedule_built = True

        _expov = random.expovariate
        _sched = self.schedule_event
        _now = self.current_time_years
        _DR = self.DEATH_CHECK_RATE
        _PR = self.PARTNER_DESIRE_RATE
        _CR = self.CHILD_DESIRE_RATE

        for person_id in list(self.alive_ids):
            person = self.people[person_id]
            age = person.age_years

            _sched(_now + _expov(_DR), EVT_DEATH, target_id=person.id)
            if age >= 12:
                _sched(_now + _expov(_PR), EVT_PARTNER_DESIRE, target_id=person.id)
                _sched(_now + _expov(_CR), EVT_CHILD_DESIRE, target_id=person.id)

        _sched(_now + _expov(self._couple_formation_rate()), EVT_COUPLE_FORMATION)

    def run(self, duration_years: float) -> None:
        """Run the simulation in continuous time for duration_years."""
        self.build_initial_schedule()
        end_time = self.current_time_years + max(0.0, duration_years)

        # Local references for the hot loop.
        _heappop = heapq.heappop
        _heappush = heapq.heappush
        _dispatch = self._dispatch
        _set_time = Person.set_sim_time
        _fel = self.fel

        while _fel and self.current_time_years < end_time:
            event = _heappop(_fel)
            evt_time = event[0]
            if evt_time > end_time:
                _heappush(_fel, event)
                break

            # Update both the simulator time and the shared Person clock.
            self.current_time_years = evt_time
            _set_time(evt_time)

            # Dispatch directly without building extra objects.
            handler = _dispatch.get(event[2])
            if handler:
                handler(event)

        if self.current_time_years < end_time:
            self.current_time_years = end_time
            _set_time(end_time)

    # ========================================================================
    # Event handlers
    # ========================================================================

    def _handle_death_event(self, event) -> None:
        target_id = event[3]
        if target_id < 0:
            return
        person = self.people.get(target_id)
        if not person or not person.is_alive:
            return

        age = person.age_years
        if age > 125:
            self._kill_person(person, "Died of extreme old age")
            return

        # Fast probability lookup.
        if person.sex == "M":
            annual_prob = self._fast_lookup(self._death_male, age)
        else:
            annual_prob = self._fast_lookup(self._death_female, age)

        delta_years = self.current_time_years - person.last_mortality_check_time
        person.last_mortality_check_time = self.current_time_years

        if delta_years > 0 and annual_prob > 0:
            period_prob = 1.0 - math.pow(1.0 - annual_prob, delta_years)
            if random.random() < period_prob:
                self._kill_person(person, "Died of natural causes")
                return

        self.schedule_event(
            self.current_time_years + random.expovariate(self.DEATH_CHECK_RATE),
            EVT_DEATH, target_id=person.id,
        )

    def _handle_partner_desire_event(self, event) -> None:
        target_id = event[3]
        if target_id < 0:
            return
        person = self.people.get(target_id)
        if not person or not person.is_alive:
            return

        age = person.age_years
        if age < 12:
            # Wait until they turn 12.
            self.schedule_event(
                self.current_time_years + max(1.0, 12.0 - age),
                EVT_PARTNER_DESIRE, target_id=person.id,
            )
            return

        person.last_partner_desire_check_time = self.current_time_years
        person.wants_partner = self._wants_partner(person)
        self._update_ready_set(person)

        self.schedule_event(
            self.current_time_years + random.expovariate(self.PARTNER_DESIRE_RATE),
            EVT_PARTNER_DESIRE, target_id=person.id,
        )

    def _handle_child_desire_event(self, event) -> None:
        target_id = event[3]
        if target_id < 0:
            return
        person = self.people.get(target_id)
        if not person or not person.is_alive:
            return

        age = person.age_years
        if age < 12:
            self.schedule_event(
                self.current_time_years + max(1.0, 12.0 - age),
                EVT_CHILD_DESIRE, target_id=person.id,
            )
            return

        person.wants_children = person.children_count < person.desired_children
        person.last_child_desire_check_time = self.current_time_years

        self.schedule_event(
            self.current_time_years + random.expovariate(self.CHILD_DESIRE_RATE),
            EVT_CHILD_DESIRE, target_id=person.id,
        )

    def _handle_couple_formation_event(self, event) -> None:
        if not self.ready_males or not self.ready_females:
            self.schedule_event(
                self.current_time_years + random.expovariate(self._couple_formation_rate()),
                EVT_COUPLE_FORMATION,
            )
            return

        attempts = max(1, min(len(self.ready_males), len(self.ready_females), 5))
        _ready_m = self.ready_males
        _ready_f = self.ready_females
        _people = self.people

        for _ in range(attempts):
            if not _ready_m or not _ready_f:
                break

            male_id = random.choice(tuple(_ready_m))
            female_id = random.choice(tuple(_ready_f))
            male = _people.get(male_id)
            female = _people.get(female_id)

            if not male or not female:
                continue
            if not male.is_alive or not female.is_alive:
                _ready_m.discard(male_id)
                _ready_f.discard(female_id)
                continue
            if male.partner_id is not None or female.partner_id is not None:
                self._update_ready_set(male)
                self._update_ready_set(female)
                continue

            m_age = male.age_years
            f_age = female.age_years
            base_prob = self._match_probability(m_age, f_age)
            noise = random.uniform(0.85, 1.15)
            prob = min(1.0, base_prob * noise)
            if random.random() < prob:
                self._form_couple(male, female)

        self.schedule_event(
            self.current_time_years + random.expovariate(self._couple_formation_rate()),
            EVT_COUPLE_FORMATION,
        )

    def _handle_pregnancy_attempt_event(self, event) -> None:
        pair_key_raw = event[4].get("pk")
        if not isinstance(pair_key_raw, str):
            return
        pair_key = tuple(int(x) for x in pair_key_raw.split("-"))
        relationship = self.relationships.get(pair_key)
        if relationship is None:
            return

        partner_a = self.people.get(relationship.partner_a)
        partner_b = self.people.get(relationship.partner_b)
        if not partner_a or not partner_b:
            return
        if not partner_a.is_alive or not partner_b.is_alive:
            return

        if partner_a.sex == "F":
            female, male = partner_a, partner_b
        else:
            female, male = partner_b, partner_a

        if female.is_pregnant or not (female.wants_children and male.wants_children):
            self._reschedule_pregnancy_attempt(relationship)
            return

        age = female.age_years
        annual_prob = self._fast_lookup(self._pregnancy, age)
        delta_years = self.current_time_years - relationship.last_pregnancy_attempt_time
        relationship.last_pregnancy_attempt_time = self.current_time_years

        if delta_years > 0 and annual_prob > 0:
            period_prob = 1.0 - math.pow(1.0 - annual_prob, delta_years)
            if random.random() < period_prob:
                self.schedule_event(
                    self.current_time_years,
                    EVT_PREGNANCY_START,
                    target_id=female.id,
                    payload={"fid": male.id},
                )

        self._reschedule_pregnancy_attempt(relationship)

    def _reschedule_pregnancy_attempt(self, relationship: RelationshipRecord) -> None:
        pk = self._pair_key(relationship.partner_a, relationship.partner_b)
        self.schedule_event(
            self.current_time_years + random.expovariate(self.PREGNANCY_ATTEMPT_RATE),
            EVT_PREGNANCY_ATTEMPT,
            payload={"pk": f"{pk[0]}-{pk[1]}"},
        )

    def _handle_pregnancy_start_event(self, event) -> None:
        target_id = event[3]
        if target_id < 0:
            return
        mother = self.people.get(target_id)
        if not mother or not mother.is_alive or mother.is_pregnant:
            return

        father_id = event[4].get("fid")
        if not isinstance(father_id, int):
            father_id = None

        num_babies = tables.babies_distribution.sample()
        mother.is_pregnant = True

        for _ in range(num_babies):
            sex = "M" if random.random() < 0.5 else "F"
            gestation_years = random.uniform(7, 10) / 12.0

            # The fetus starts with a negative age and is not alive yet.
            fetus = Person(sex, -gestation_years, is_alive=False)
            fetus.desired_children = tables.desired_children_distribution.sample()
            self.people[fetus.id] = fetus

            due = self.current_time_years + gestation_years
            self.gestation_group[fetus.id] = GestationRecord(
                fetus_id=fetus.id,
                mother_id=mother.id,
                father_id=father_id,
                due_time=due,
            )
            self.mother_gestations.setdefault(mother.id, set()).add(fetus.id)

            # Births only happen through the birth event.
            self.schedule_event(due, EVT_BIRTH, target_id=fetus.id)

    def _handle_birth_event(self, event) -> None:
        target_id = event[3]
        if target_id < 0:
            return
        if target_id not in self.gestation_group:
            return
        self._finalize_birth(target_id)

    def _finalize_birth(self, fetus_id: int) -> None:
        """Move a fetus into the active population and clear gestation state."""
        record = self.gestation_group.pop(fetus_id, None)
        if record is None:
            return
        fetus = self.people[fetus_id]

        # Reset the newborn age to zero at the current time.
        fetus.age_years = 0.0
        fetus.is_alive = True
        fetus.last_mortality_check_time = self.current_time_years
        fetus.last_partner_desire_check_time = self.current_time_years
        fetus.last_child_desire_check_time = self.current_time_years

        self._add_person_to_population(fetus)
        self.total_births += 1

        mother = self.people.get(record.mother_id)
        if mother and mother.is_alive:
            mother.children_count += 1
        if record.father_id is not None:
            father = self.people.get(record.father_id)
            if father and father.is_alive:
                father.children_count += 1

        if mother and mother.id in self.mother_gestations:
            self.mother_gestations[mother.id].discard(fetus_id)
            if not self.mother_gestations[mother.id]:
                mother.is_pregnant = False
                self.mother_gestations.pop(mother.id, None)

        self._log_event(f"Birth: {fetus}")

        # Schedule the newborn's first events.
        self.schedule_event(
            self.current_time_years + random.expovariate(self.DEATH_CHECK_RATE),
            EVT_DEATH, target_id=fetus.id,
        )
        # Delay partner and child desire checks until age 12.
        self.schedule_event(
            self.current_time_years + 12.0,
            EVT_PARTNER_DESIRE, target_id=fetus.id,
        )
        self.schedule_event(
            self.current_time_years + 12.0,
            EVT_CHILD_DESIRE, target_id=fetus.id,
        )

    def _handle_breakup_event(self, event) -> None:
        pair_key_raw = event[4].get("pk")
        if not isinstance(pair_key_raw, str):
            return
        pair_key = tuple(int(x) for x in pair_key_raw.split("-"))
        relationship = self.relationships.get(pair_key)
        if relationship is None:
            return

        partner_a = self.people.get(relationship.partner_a)
        partner_b = self.people.get(relationship.partner_b)
        if not partner_a or not partner_b:
            return
        if not partner_a.is_alive or not partner_b.is_alive:
            return

        annual_breakup = 0.20  # Fixed yearly breakup chance.
        delta_years = self.current_time_years - relationship.last_breakup_check_time
        relationship.last_breakup_check_time = self.current_time_years

        if delta_years > 0 and annual_breakup > 0:
            period_prob = 1.0 - math.pow(1.0 - annual_breakup, delta_years)
            if random.random() < period_prob:
                self._dissolve_relationship(pair_key, "Couple breakup")
                return

        self.schedule_event(
            self.current_time_years + random.expovariate(self.BREAKUP_CHECK_RATE),
            EVT_BREAKUP,
            payload={"pk": f"{pair_key[0]}-{pair_key[1]}"},
        )

    # ========================================================================
    # Relationship management
    # ========================================================================

    def _form_couple(self, male: Person, female: Person) -> None:
        male.partner_id = female.id
        female.partner_id = male.id
        male.wants_partner = False
        female.wants_partner = False
        self._update_ready_set(male)
        self._update_ready_set(female)
        self.total_couples_formed += 1

        pair_key = self._pair_key(male.id, female.id)
        self.relationships[pair_key] = RelationshipRecord(
            partner_a=male.id, partner_b=female.id,
            last_breakup_check_time=self.current_time_years,
            last_pregnancy_attempt_time=self.current_time_years,
        )

        self._log_event(f"New couple: {male} and {female}")

        self.schedule_event(
            self.current_time_years + random.expovariate(self.BREAKUP_CHECK_RATE),
            EVT_BREAKUP,
            payload={"pk": f"{pair_key[0]}-{pair_key[1]}"},
        )
        self._reschedule_pregnancy_attempt(self.relationships[pair_key])

    def _dissolve_relationship(self, pair_key: Tuple[int, int], reason: str) -> None:
        relationship = self.relationships.pop(pair_key, None)
        if relationship is None:
            return

        self.total_breakups += 1
        partner_a = self.people.get(relationship.partner_a)
        partner_b = self.people.get(relationship.partner_b)
        if partner_a and partner_b:
            partner_a.partner_id = None
            partner_b.partner_id = None

            # Store the grief end time directly.
                        # Store the grief end time directly.
            partner_a.grief_end_time = self.current_time_years + self._get_grief_time_years(partner_a.age_years)
            partner_b.grief_end_time = self.current_time_years + self._get_grief_time_years(partner_b.age_years)

            self._update_ready_set(partner_a)
            self._update_ready_set(partner_b)
            self._log_event(f"{reason} between {partner_a} and {partner_b}")

    def _update_ready_set(self, person: Person) -> None:
        if person.partner_id is not None or not person.wants_partner or not person.is_alive:
            self.ready_males.discard(person.id)
            self.ready_females.discard(person.id)
            return
        if person.sex == "M":
            self.ready_males.add(person.id)
        else:
            self.ready_females.add(person.id)

    def _kill_person(self, person: Person, reason: str) -> None:
        if not person.is_alive:
            return
        person.is_alive = False
        self.total_deaths += 1
        self.alive_ids.discard(person.id)
        self.dead_ids.add(person.id)
        self.ready_males.discard(person.id)
        self.ready_females.discard(person.id)
        if person.sex == "M":
            self.male_group.discard(person.id)
        else:
            self.female_group.discard(person.id)

        if person.partner_id is not None:
            pair_key = self._pair_key(person.id, person.partner_id)
            self._dissolve_relationship(pair_key, "Couple ends due to death")

        self._log_event(f"{reason}: {person}")

    def tick(self) -> None:
        """Advance the simulation by one month."""
        self.run(1.0 / 12.0)

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_statistics(self) -> Dict:
        alive = self.population_alive
        n_alive = len(alive)
        if n_alive == 0:
            return {
                "time_years": round(self.current_time_years, 2),
                "population": 0, "males": 0, "females": 0,
                "avg_age": 0, "births": self.total_births,
                "deaths": self.total_deaths, "couples": len(self.relationships),
                "pregnant": 0, "breakups": self.total_breakups,
            }

        males = sum(1 for p in alive if p.sex == "M")
        females = n_alive - males
        ages = [p.age_years for p in alive]

        return {
            "time_years": round(self.current_time_years, 2),
            "population": n_alive,
            "males": males,
            "females": females,
            "avg_age": round(sum(ages) / n_alive, 1),
            "max_age": round(max(ages), 1),
            "min_age": round(min(ages), 1),
            "births": self.total_births,
            "deaths": self.total_deaths,
            "couples": len(self.relationships),
            "pregnant": sum(1 for p in alive if p.is_pregnant),
            "breakups": self.total_breakups,
            "couples_formed": self.total_couples_formed,
        }

    def count_couples(self) -> int:
        return len(self.relationships)

    def get_age_distribution(self, bins=None) -> Dict[str, int]:
        if bins is None:
            bins = [0, 12, 25, 45, 65, 125]
        alive = self.population_alive
        dist = {}
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            label = f"{int(lo)}-{int(hi)}"
            dist[label] = sum(1 for p in alive if lo <= p.age_years < hi)
        dist["65+"] = sum(1 for p in alive if p.age_years >= 65)
        if "65-125" in dist:
            del dist["65-125"]
        return dist
